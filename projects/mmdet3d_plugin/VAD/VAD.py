import time
import copy

import torch
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmcv.runner import force_fp32, auto_fp16
from scipy.optimize import linear_sum_assignment
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric


# ============================================================================
#  VAD 模型主入口 (VAD.py 总览)
# ----------------------------------------------------------------------------
#  数据流（一次前向）：
#    多视角图像 (B, N=6, C, H, W)
#        │  img_backbone (ResNet50/101)  → 多尺度 2D 特征
#        ▼  img_neck (FPN)
#    img_feats: list[Tensor]  形如 (B, N, C, H', W')
#        │
#        │  pts_bbox_head (= VADHead) 内部:
#        │    1) BEV encoder（BEVFormer 风格）：cross-attention
#        │       把多视角图像特征聚合到 BEV 网格上 → bev_embed (B, H*W, C)
#        │    2) Object detection head：3D 框 + 他车未来轨迹（motion）
#        │    3) Map head：向量化车道线 / 边界 polyline
#        │    4) Planning head：自车未来 3 秒轨迹（v1 回归 / v2 词表分类）
#        ▼
#    outs (dict): {bbox_pred, traj_pred, map_pred, ego_fut_preds, bev_embed, ...}
#
#  关键概念：
#    - prev_bev：上一帧的 BEV 特征，用作时序输入（temporal self-attention）
#    - ego_his_trajs / ego_fut_trajs：自车历史 / 未来轨迹（GT 或预测）
#    - ego_fut_cmd：高层指令（左转 / 右转 / 直行），one-hot 编码
#    - ego_lcf_feat：自车低频运动特征（速度、加速度、转向角等）
#
#  本文件 (VAD.py) 只是"调度员"，真正的网络计算在 VAD_head.py 里。
# ============================================================================


@DETECTORS.register_module()
class VAD(MVXTwoStageDetector):
    """VAD model.

    继承自 mmdet3d 的 MVXTwoStageDetector（多视角融合两阶段检测器基类）。
    VAD 复用了它的 img_backbone / img_neck 等组件，但把 pts_bbox_head 替换成
    VADHead（自定义的端到端 head，包含感知 + 地图 + 规划）。
    """
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 fut_ts=6,
                 fut_mode=6
                 ):
        # 构造函数：父类已经把 backbone / neck / head 都创建好了
        # 重要参数：
        #   img_backbone / img_neck → 图像特征提取（ResNet + FPN）
        #   pts_bbox_head           → VADHead，包含 BEV encoder + 三个任务头
        #   fut_ts = 6              → 预测未来 6 个时间步（每步 0.5s，共 3 秒）
        #   fut_mode = 6            → 他车多模态预测的模态数
        #   video_test_mode         → 测试时是否使用时序信息（上一帧 BEV）
        super(VAD,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        # GridMask：训练时的图像数据增强（随机遮挡网格区域）
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.valid_fut_ts = pts_bbox_head['valid_fut_ts']

        # 时序状态：测试时跨帧传递 BEV 特征和 ego 位姿
        # 同一个 scene 内累积，遇到新 scene 就重置
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,       # 上一帧的 BEV 特征
            'scene_token': None,    # 当前 scene 标识，用来判断是否换场景
            'prev_pos': 0,          # 上一帧 ego 位置
            'prev_angle': 0,        # 上一帧 ego 朝向
        }

        self.planning_metric = None  # 延迟初始化：第一次评测时再建

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """从多视角图像提取 2D 特征 —— 这是数据流的第一步。

        输入 img 形状：
          - 训练时带时序：(B, len_queue, N=6, C, H, W)
          - 单帧：       (B, N=6, C, H, W) 或 (1, B*N, C, H, W)
        其中 N=6 = 前/前左/前右/后/后左/后右 6 个相机。

        步骤：
          1. 把 (B, N, C, H, W) reshape 成 (B*N, C, H, W) 喂给 2D backbone
          2. 可选 GridMask 增强
          3. img_backbone (ResNet) → img_neck (FPN) → 多尺度特征
          4. 再 reshape 回 (B, N, C, H', W') 形式，方便后续按 batch 处理

        返回：list of Tensor，每个对应 FPN 的一个尺度
        """
        B = img.size(0)
        if img is not None:

            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            # 处理不同的输入维度：可能带时序（5D）也可能不带
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)  # 把 batch 和相机数合并
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)  # ResNet 输出多尺度特征
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)  # FPN 融合多尺度

        # 把每个尺度的特征 reshape 回带 batch / 相机维度的形状
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                # 带时序：(B, len_queue, N, C, H, W)
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                # 单帧：(B, N, C, H, W)
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          map_gt_bboxes_3d,
                          map_gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          map_gt_bboxes_ignore=None,
                          prev_bev=None,
                          ego_his_trajs=None,
                          ego_fut_trajs=None,
                          ego_fut_masks=None,
                          ego_fut_cmd=None,
                          ego_lcf_feat=None,
                          gt_attr_labels=None):
        """训练时调用：跑一次 head 前向 + 计算 loss。

        这是【训练分支】最核心的函数，干两件事：
          1. self.pts_bbox_head(...)         → 跑出预测 outs（含 BEV、检测、地图、规划）
          2. self.pts_bbox_head.loss(...)    → 把 outs 和 GT 对比算 loss

        loss 包含的几大类：
          - det loss        : 3D 检测（cls + bbox 回归）
          - motion loss     : 他车多模态未来轨迹
          - map loss        : 向量化地图（cls + 点回归）
          - planning loss   : 自车未来轨迹（轨迹回归 + 三个约束）
              · 与他车碰撞约束
              · 越过车道边界约束
              · 行驶方向约束
        """

        # 跑 head：内部依次完成 BEV 编码 → 三个任务 head 解码
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
        # 把所有 GT 和预测打包，丢给 head.loss 一次性算所有 loss
        loss_inputs = [
            gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d, map_gt_labels_3d,
            outs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd, gt_attr_labels
        ]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """模型总入口。
        - return_loss=True  → 训练分支 forward_train，返回 loss 字典
        - return_loss=False → 测试分支 forward_test，返回检测/规划结果

        Note this setting will change the expected inputs.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """对历史多帧图像逐帧跑一遍，得到上一帧的 BEV 特征。

        关键点：
          - self.eval() + torch.no_grad()：历史帧不算梯度，省显存
          - 一次性把整个 queue 过 backbone（节省时间），再逐帧过 BEV encoder
          - 返回的 prev_bev 会作为当前帧 BEV encoder 的时序输入
            （在 BEVFormer 的 temporal self-attention 里和当前 BEV query 做对齐）
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                # only_bev=True：只跑 BEV encoder，不跑后续三个 head
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None
                      ):
        """训练分支主流程。

        关键 GT 输入（端到端的"多任务监督"）：
          - gt_bboxes_3d / gt_labels_3d         : 3D 检测 GT
          - map_gt_bboxes_3d / map_gt_labels_3d : 向量化地图 GT（车道线、边界、人行道）
          - gt_attr_labels                      : 包含他车未来轨迹 GT
          - ego_fut_trajs / ego_fut_masks       : 自车未来轨迹 GT
          - ego_his_trajs                       : 自车历史轨迹（输入到 planning head）
          - ego_fut_cmd                         : 高层指令 one-hot（左/右/直）
          - ego_lcf_feat                        : 自车低频特征（速度/加速度/转角等）

        三步走：
          1) 拆分时序：img 形状 (B, len_queue, N=6, C, H, W)
                       prev_img 是历史帧，img 是当前帧
          2) 历史帧过 obtain_history_bev → prev_bev（无梯度）
          3) 当前帧过 extract_feat + forward_pts_train → 出 loss
        """

        # 步骤 1：拆出历史帧和当前帧
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]   # 前 len_queue-1 帧（历史）
        img = img[:, -1, ...]          # 最后一帧（当前）

        prev_img_metas = copy.deepcopy(img_metas)
        # 步骤 2：跑历史 BEV（仅当有历史帧时）
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue > 1 else None

        # 取出当前帧的 metas
        img_metas = [each[len_queue-1] for each in img_metas]
        # 步骤 3：当前帧图像 → 2D 特征
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        # head 前向 + loss
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d,
                                            map_gt_bboxes_3d, map_gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, map_gt_bboxes_ignore, prev_bev,
                                            ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
                                            ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
                                            ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels)

        losses.update(losses_pts)
        return losses

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        """测试分支主流程。

        与训练的关键区别：
          - 不算 loss，只算预测结果 + 评测指标
          - 时序信息靠 self.prev_frame_info 跨调用维护（而不是一次塞 queue）
          - 跨场景边界自动重置 prev_bev
          - can_bus 信息（位置 / 朝向）做相对化：用 当前 - 上一帧 的 delta
        """
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        # 场景切换检测：进入新 scene 时清空 prev_bev
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # 关闭时序：每帧都从头算 BEV（用于消融实验）
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # 计算 ego 位姿的相对 delta（喂给 BEV encoder 做时序对齐）
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            # 有上一帧：用相对位移 / 相对角度
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            # 新场景第一帧：清零，没有相对参考
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        # 实际推理
        new_prev_bev, bbox_results = self.simple_test(
            img_metas=img_metas[0],
            img=img[0],
            prev_bev=self.prev_frame_info['prev_bev'],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0],
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0],
            gt_attr_labels=gt_attr_labels,
            **kwargs
        )
        # 保存当前帧状态，给下一帧用
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev

        return bbox_results

    def simple_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        prev_bev=None,
        points=None,
        fut_valid_flag=None,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts, metric_dict = self.simple_test_pts(
            img_feats,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            prev_bev,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            start=None,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict

        return new_prev_bev, bbox_list

    def simple_test_pts(
        self,
        x,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        prev_bev=None,
        fut_valid_flag=None,
        rescale=False,
        start=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
    ):
        """单帧测试 + 评测计算（最重要的"出结果"函数）。

        三大产出：
          1. 检测/规划预测结果 bbox_results（含 trajs_3d、ego_fut_preds、map_pts_3d 等）
          2. motion 评测指标（ADE / FDE / MR / EPA）—— 他车多模态预测
          3. planning 评测指标（L2 / 碰撞率）       —— 自车规划

        关键流程：
          - head 前向 → outs
          - get_bboxes 解码 → list of (检测框、轨迹、地图点 ...)
          - 用阈值 0.6 过滤低置信度框
          - 通过匈牙利匹配把预测和 GT 配对（assign_pred_to_gt_vip3d）
          - 按 ego_fut_cmd 选出对应模态的 ego 轨迹（v1 是 cmd 条件回归）
          - 累加（cumsum）成全局轨迹后再算 L2 和碰撞
        """
        mapped_class_names = [
            'car', 'truck', 'construction_vehicle', 'bus',
            'trailer', 'barrier', 'motorcycle', 'bicycle',
            'pedestrian', 'traffic_cone'
        ]

        # head 前向：输出感知 + 地图 + 规划
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev,
                                  ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
        # 解码 outs → 实际可用的框、轨迹、地图点
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = []
        for i, (bboxes, scores, labels, trajs, map_bboxes, \
                map_scores, map_labels, map_pts) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result['trajs_3d'] = trajs.cpu()                 # 他车未来轨迹（多模态）
            map_bbox_result = self.map_pred2result(map_bboxes, map_scores, map_labels, map_pts)
            bbox_result.update(map_bbox_result)                   # 地图预测：边界 + 点序列
            bbox_result['ego_fut_preds'] = outs['ego_fut_preds'][i].cpu()  # 自车未来轨迹（v1: 3 个 cmd 各一条）
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, 'only support batch_size=1 now'
        score_threshold = 0.6
        with torch.no_grad():
            c_bbox_results = copy.deepcopy(bbox_results)

            bbox_result = c_bbox_results[0]
            gt_bbox = gt_bboxes_3d[0][0]
            gt_label = gt_labels_3d[0][0].to('cpu')
            gt_attr_label = gt_attr_labels[0][0].to('cpu')
            fut_valid_flag = bool(fut_valid_flag[0][0])
            # 过滤低置信度框
            mask = bbox_result['scores_3d'] > score_threshold
            bbox_result['boxes_3d'] = bbox_result['boxes_3d'][mask]
            bbox_result['scores_3d'] = bbox_result['scores_3d'][mask]
            bbox_result['labels_3d'] = bbox_result['labels_3d'][mask]
            bbox_result['trajs_3d'] = bbox_result['trajs_3d'][mask]

            # 把 pred 和 GT 用匈牙利算法匹配（按中心距离）
            matched_bbox_result = self.assign_pred_to_gt_vip3d(
                bbox_result, gt_bbox, gt_label)

            # 算 motion 评测：ADE / FDE / MR / fp / hit
            metric_dict = self.compute_motion_metric_vip3d(
                gt_bbox, gt_label, gt_attr_label, bbox_result,
                matched_bbox_result, mapped_class_names)

            # ego planning 评测
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            ego_fut_preds = bbox_result['ego_fut_preds']
            ego_fut_trajs = ego_fut_trajs[0, 0]
            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            # 根据高层指令（左/右/直）选对应那条预测轨迹
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0, 0]
            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]
            # 网络输出的是"位移增量"，cumsum 后才是实际轨迹点
            ego_fut_pred = ego_fut_pred.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            # 算 L2 和碰撞率（与 ST-P3 一致的评测口径）
            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_pred[None],
                gt_ego_fut_trajs = ego_fut_trajs[None],
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label.unsqueeze(0),
                fut_valid_flag = fut_valid_flag
            )
            metric_dict.update(metric_dict_planner_stp3)

        return outs['bev_embed'], bbox_results, metric_dict

    def map_pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            map_boxes_3d=bboxes.to('cpu'),
            map_scores_3d=scores.cpu(),
            map_labels_3d=labels.cpu(),
            map_pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['map_attrs_3d'] = attrs.cpu()

        return result_dict

    def assign_pred_to_gt_vip3d(
        self,
        bbox_result,
        gt_bbox,
        gt_label,
        match_dis_thresh=2.0
    ):
        """Assign pred boxs to gt boxs according to object center preds in lcf.
        Args:
            bbox_result (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
        """     
        dynamic_list = [0,1,3,4,6,7,8]
        matched_bbox_result = torch.ones(
            (len(gt_bbox)), dtype=torch.long) * -1  # -1: not assigned
        gt_centers = gt_bbox.center[:, :2]
        pred_centers = bbox_result['boxes_3d'].center[:, :2]
        dist = torch.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], dim=-1)
        pred_not_dyn = [label not in dynamic_list for label in bbox_result['labels_3d']]
        gt_not_dyn = [label not in dynamic_list for label in gt_label]
        dist[pred_not_dyn] = 1e6
        dist[:, gt_not_dyn] = 1e6
        dist[dist > match_dis_thresh] = 1e6

        r_list, c_list = linear_sum_assignment(dist)

        for i in range(len(r_list)):
            if dist[r_list[i], c_list[i]] <= match_dis_thresh:
                matched_bbox_result[c_list[i]] = r_list[i]

        return matched_bbox_result

    def compute_motion_metric_vip3d(
        self,
        gt_bbox,
        gt_label,
        gt_attr_label,
        pred_bbox,
        matched_bbox_result,
        mapped_class_names,
        match_dis_thresh=2.0,
    ):
        """Compute EPA metric for one sample.
        Args:
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            pred_bbox (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            EPA_dict (dict): EPA metric dict of each cared class.
        """
        motion_cls_names = ['car', 'pedestrian']
        motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',
                               'fp', 'ADE', 'FDE', 'MR']
        
        metric_dict = {}
        for met in motion_metric_names:
            for cls in motion_cls_names:
                metric_dict[met+'_'+cls] = 0.0

        veh_list = [0,1,3,4]
        ignore_list = ['construction_vehicle', 'barrier',
                       'traffic_cone', 'motorcycle', 'bicycle']

        for i in range(pred_bbox['labels_3d'].shape[0]):
            pred_bbox['labels_3d'][i] = 0 if pred_bbox['labels_3d'][i] in veh_list else pred_bbox['labels_3d'][i]
            box_name = mapped_class_names[pred_bbox['labels_3d'][i]]
            if box_name in ignore_list:
                continue
            if i not in matched_bbox_result:
                metric_dict['fp_'+box_name] += 1

        for i in range(gt_label.shape[0]):
            gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
            box_name = mapped_class_names[gt_label[i]]
            if box_name in ignore_list:
                continue
            gt_fut_masks = gt_attr_label[i][self.fut_ts*2:self.fut_ts*3]
            num_valid_ts = sum(gt_fut_masks==1)
            if num_valid_ts == self.fut_ts:
                metric_dict['gt_'+box_name] += 1
            if matched_bbox_result[i] >= 0 and num_valid_ts > 0:
                metric_dict['cnt_ade_'+box_name] += 1
                m_pred_idx = matched_bbox_result[i]
                gt_fut_trajs = gt_attr_label[i][:self.fut_ts*2].reshape(-1, 2)
                gt_fut_trajs = gt_fut_trajs[:num_valid_ts]
                pred_fut_trajs = pred_bbox['trajs_3d'][m_pred_idx].reshape(self.fut_mode, self.fut_ts, 2)
                pred_fut_trajs = pred_fut_trajs[:, :num_valid_ts, :]
                gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2)
                pred_fut_trajs = pred_fut_trajs.cumsum(dim=-2)
                gt_fut_trajs = gt_fut_trajs + gt_bbox[i].center[0, :2]
                pred_fut_trajs = pred_fut_trajs + pred_bbox['boxes_3d'][int(m_pred_idx)].center[0, :2]

                dist = torch.linalg.norm(gt_fut_trajs[None, :, :] - pred_fut_trajs, dim=-1)
                ade = dist.sum(-1) / num_valid_ts
                ade = ade.min()

                metric_dict['ADE_'+box_name] += ade
                if num_valid_ts == self.fut_ts:
                    fde = dist[:, -1].min()
                    metric_dict['cnt_fde_'+box_name] += 1
                    metric_dict['FDE_'+box_name] += fde
                    if fde <= match_dis_thresh:
                        metric_dict['hit_'+box_name] += 1
                    else:
                        metric_dict['MR_'+box_name] += 1

        return metric_dict

    ### same planning metric as stp3
    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag
    ):
        """计算自车规划评测指标（与 ST-P3 论文一致）—— 面试常考！

        两类指标：
          1. plan_L2_{1,2,3}s        : 1/2/3 秒处的 L2 误差（单位：米）
          2. plan_obj_col_{1,2,3}s   : 1/2/3 秒处的碰撞率（基于 occupancy 像素）
             plan_obj_box_col_{1,2,3}s : 基于 box（更严格）

        关键步骤：
          1. PlanningMetric 把他车 GT box 渲染成 BEV occupancy 图
          2. 把自车未来轨迹"踩"到 occupancy 上，看是否碰到他车占用区
          3. fut_valid_flag=False 表示这帧 GT 不全（如场景边缘），跳过

        【面试加分点】这种开环评测的局限性：
          - 不闭环：模型预测的轨迹不会真的让车动，下一帧仍是 GT 位置 → 误差不积累
          - 不交互：他车按 GT 轨迹走，不会响应自车
          - L2 偏向"模仿专家"，不一定代表安全 / 舒适 / 通行效率
          - 业界后续转向 CARLA / nuPlan 闭环、RAD（强化学习闭环）
        """
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        # 把他车 GT 渲染成 BEV occupancy mask
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)

        # 分别算 1s / 2s / 3s 的指标（每秒 2 个时间步，cur_time = 2/4/6）
        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                # L2：欧氏距离平均
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                # 碰撞：obj_coll 是简单点碰撞，obj_box_coll 是把自车按车体 box 检查
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.mean().item()
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0

        return metric_dict

    def set_epoch(self, epoch): 
        self.pts_bbox_head.epoch = epoch