import os
import math
import copy
import argparse
from os import path as osp
from collections import OrderedDict
from typing import List, Tuple, Union

import mmcv
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from shapely.geometry import MultiPoint, box
from mmdet3d.datasets import NuScenesDataset
from nuscenes.utils.geometry_utils import view_points
from mmdet3d.core.bbox.box_np_ops import points_cam2img
from nuscenes.utils.geometry_utils import transform_matrix


nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

ego_width, ego_length = 1.85, 4.084

def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i


def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    print(version, root_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(out_path,
                                 '{}_infos_temporal_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)


def _fill_trainval_infos(nusc,
                         nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         fut_ts=6,
                         his_ts=2):
    """
    【数据流第1层：预处理脚本 - 最源头】
    生成训练/验证集的信息字典，这是 VAD 训练数据的起点。

    功能：从 nuScenes 原始数据生成 .pkl 文件，包含所有训练所需的 GT。

    核心工作（对应文档的5大类数据）：
      1. 读取6相机图像路径、标定参数（类别1：图像特征的源头）
      2. 读取3D检测框标注（类别2：人工标注）
      3. 读取HD Map并矢量化（类别3：地图GT）
      4. 计算ego历史/未来轨迹、指令、低频特征（类别4：免费午餐⭐）
      5. 追踪并生成agent未来轨迹（类别5：追踪算法）

    Args:
        nusc (:obj:`NuScenes`): nuScenes 数据集对象
        nusc_can_bus: CAN 总线数据（用于ego_lcf_feat）
        train_scenes (list[str]): 训练场景列表
        val_scenes (list[str]): 验证场景列表
        test (bool): 是否测试模式（无标注）
        max_sweeps (int): 最大sweep数量（LiDAR）
        fut_ts (int): 未来时间步数（默认6，即3秒@2Hz）
        his_ts (int): 历史时间步数（默认2，即1秒@2Hz，实际用4帧=2秒）

    Returns:
        tuple[list[dict]]: 训练集和验证集的信息列表
            每个 dict 包含一个 sample 的完整信息（对应一个训练样本）
    """
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    cat2idx = {}
    for idx, dic in enumerate(nusc.category):
        cat2idx[dic['name']] = idx

    for sample in mmcv.track_iter_progress(nusc.sample):
        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        if sample['prev'] != '':
            sample_prev = nusc.get('sample', sample['prev'])
            sd_rec_prev = nusc.get('sample_data', sample_prev['data']['LIDAR_TOP'])
            pose_record_prev = nusc.get('ego_pose', sd_rec_prev['ego_pose_token'])
        else:
            pose_record_prev = None
        if sample['next'] != '':
            sample_next = nusc.get('sample', sample['next'])
            sd_rec_next = nusc.get('sample_data', sample_next['data']['LIDAR_TOP'])
            pose_record_next = nusc.get('ego_pose', sd_rec_next['ego_pose_token'])
        else:
            pose_record_next = None

        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)
        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)
        fut_valid_flag = True
        test_sample = copy.deepcopy(sample)
        for i in range(fut_ts):
            if test_sample['next'] != '':
                test_sample = nusc.get('sample', test_sample['next'])
            else:
                fut_valid_flag = False
        ##
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'can_bus': can_bus,
            'frame_idx': frame_idx,  # temporal related info
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],  # temporal related info
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'fut_valid_flag': fut_valid_flag,
            'map_location': map_location
        }

        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            # ============================================================
            # 【类别2：3D检测GT - 人工标注，高成本】
            # 从 nuScenes 的人工标注读取 3D 检测框
            # ============================================================
            # 获取当前帧的所有标注（annotations）
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            # 提取3D框的中心、尺寸、朝向
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)  # 中心 (x,y,z)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)      # 尺寸 (w,l,h)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)            # 朝向 yaw
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])  # 速度 (vx,vy)
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)  # 过滤点云数为0的框（遮挡严重）

            # 将速度从 global 坐标系转换到 lidar 坐标系
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            # 类别名称映射（nuScenes → VAD）
            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # 转换为 SECOND 格式：[x,y,z, w,l,h, yaw]
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'

            # ============================================================
            # 【类别5：他车未来轨迹GT - 追踪算法生成】
            # 通过追踪每个 agent 的 instance_token，查询未来帧的位置
            # ============================================================
            # get future coords for each box
            # [num_box, fut_ts*2]
            num_box = len(boxes)
            gt_fut_trajs = np.zeros((num_box, fut_ts, 2))   # 未来6步轨迹 [N_obj, 6, 2]
            gt_fut_yaw = np.zeros((num_box, fut_ts))        # 未来6步朝向 [N_obj, 6]
            gt_fut_masks = np.zeros((num_box, fut_ts))      # 有效性掩码 [N_obj, 6]
            gt_boxes_yaw = -(gt_boxes[:,6] + np.pi / 2)     # 当前朝向角
            # agent lcf feat (x, y, yaw, vx, vy, width, length, height, type)
            agent_lcf_feat = np.zeros((num_box, 9))         # agent 低频特征 [N_obj, 9]
            gt_fut_goal = np.zeros((num_box))               # 终点方向分类 [N_obj]

            # 对每个 agent 进行追踪和未来轨迹提取
            for i, anno in enumerate(annotations):
                cur_box = boxes[i]
                cur_anno = anno
                # 记录 agent 当前状态到 lcf_feat
                agent_lcf_feat[i, 0:2] = cur_box.center[:2]	  # 当前位置 (x,y)
                agent_lcf_feat[i, 2] = gt_boxes_yaw[i]        # 当前朝向 yaw
                agent_lcf_feat[i, 3:5] = velocity[i]          # 当前速度 (vx,vy)
                agent_lcf_feat[i, 5:8] = anno['size']         # 尺寸 (w,l,h)
                agent_lcf_feat[i, 8] = cat2idx[anno['category_name']] if anno['category_name'] in cat2idx.keys() else -1  # 类别

                # 往后追踪 fut_ts=6 步（未来3秒）
                for j in range(fut_ts):
                    if cur_anno['next'] != '':  # 还有下一帧
                        # 通过 instance_token 关联同一物体在下一帧的标注
                        anno_next = nusc.get('sample_annotation', cur_anno['next'])
                        box_next = Box(
                            anno_next['translation'], anno_next['size'], Quaternion(anno_next['rotation'])
                        )
                        # 坐标变换：global → ego vehicle → lidar（与当前帧对齐）
                        box_next.translate(-np.array(pose_record['translation']))
                        box_next.rotate(Quaternion(pose_record['rotation']).inverse)
                        box_next.translate(-np.array(cs_record['translation']))
                        box_next.rotate(Quaternion(cs_record['rotation']).inverse)

                        # 记录相对位移（offset格式，而非累积位置）
                        gt_fut_trajs[i, j] = box_next.center[:2] - cur_box.center[:2]
                        gt_fut_masks[i, j] = 1  # 标记此步有效

                        # 记录朝向变化
                        _, _, box_yaw = quart_to_rpy([cur_box.orientation.x, cur_box.orientation.y,
                                                      cur_box.orientation.z, cur_box.orientation.w])
                        _, _, box_yaw_next = quart_to_rpy([box_next.orientation.x, box_next.orientation.y,
                                                           box_next.orientation.z, box_next.orientation.w])
                        gt_fut_yaw[i, j] = box_yaw_next - box_yaw

                        # 移动到下一帧继续追踪
                        cur_anno = anno_next
                        cur_box = box_next
                    else:  # 场景结束，剩余时间步填0
                        gt_fut_trajs[i, j:] = 0
                        break

                # 计算终点目标（用于多模态预测的 goal-oriented）
                gt_fut_coords = np.cumsum(gt_fut_trajs[i], axis=-2)  # offset → 累积坐标
                coord_diff = gt_fut_coords[-1] - gt_fut_coords[0]   # 起点到终点的向量
                if coord_diff.max() < 1.0:  # 几乎不动，判定为静止
                    gt_fut_goal[i] = 9      # 静止类别
                else:
                    # 计算运动方向并量化为8个方向（0°-360°分8份）
                    box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
                    gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)  # 0-7: 8个方向

            # ============================================================
            # 【类别4：自车历史轨迹 - 免费午餐⭐，从ego pose日志直接计算】
            # 往前查询 his_ts+1=3 帧（实际代码中会用4帧），得到过去的轨迹
            # 关键：这是"事后回看"，不需要任何标注！
            # ============================================================
            # get ego history traj (offset format)
            ego_his_trajs = np.zeros((his_ts+1, 3))        # [3, 3] 历史位置
            ego_his_trajs_diff = np.zeros((his_ts+1, 3))  # [3, 3] 用于插值
            sample_cur = sample

            # 从当前帧往前追溯
            for i in range(his_ts, -1, -1):  # i = 2, 1, 0（倒序填充）
                if sample_cur is not None:
                    # 读取该帧的 ego pose（global 坐标系）
                    pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
                    ego_his_trajs[i] = pose_mat[:3, 3]  # 提取平移向量 (x,y,z)

                    has_prev = sample_cur['prev'] != ''
                    has_next = sample_cur['next'] != ''
                    # 计算与下一帧的位移（用于插值缺失帧）
                    if has_next:
                        sample_next = nusc.get('sample', sample_cur['next'])
                        pose_mat_next = get_global_sensor_pose(sample_next, nusc, inverse=False)
                        ego_his_trajs_diff[i] = pose_mat_next[:3, 3] - ego_his_trajs[i]

                    # 移动到上一帧
                    sample_cur = nusc.get('sample', sample_cur['prev']) if has_prev else None
                else:
                    # 如果已经到场景开头，用线性插值填充
                    ego_his_trajs[i] = ego_his_trajs[i+1] - ego_his_trajs_diff[i+1]
                    ego_his_trajs_diff[i] = ego_his_trajs_diff[i+1]

            # 坐标变换：global → ego vehicle at current frame
            ego_his_trajs = ego_his_trajs - np.array(pose_record['translation'])
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T

            # 坐标变换：ego → lidar（nuScenes的lidar是ego的一个传感器）
            ego_his_trajs = ego_his_trajs - np.array(cs_record['translation'])
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T

            # 转换为 offset 格式（每步相对于上一步的位移，而非累积坐标）
            ego_his_trajs = ego_his_trajs[1:] - ego_his_trajs[:-1]  # [2, 3] → 实际是2步历史

            # ============================================================
            # 【类别4：自车未来轨迹 - 免费午餐⭐，训练时"未来"已经发生了】
            # 往后查询 fut_ts+1=7 帧，得到未来3秒的轨迹
            # 这是数据闭环的核心优势：部署时实时采集，事后就能生成训练GT
            # ============================================================
            # get ego futute traj (offset format)
            ego_fut_trajs = np.zeros((fut_ts+1, 3))   # [7, 3] 未来位置
            ego_fut_masks = np.zeros((fut_ts+1))      # [7] 有效性掩码
            sample_cur = sample

            # 从当前帧往后查询
            for i in range(fut_ts+1):  # i = 0, 1, ..., 6
                pose_mat = get_global_sensor_pose(sample_cur, nusc, inverse=False)
                ego_fut_trajs[i] = pose_mat[:3, 3]
                ego_fut_masks[i] = 1  # 标记有效

                if sample_cur['next'] == '':  # 场景结束
                    # 剩余时间步用最后位置填充
                    ego_fut_trajs[i+1:] = ego_fut_trajs[i]
                    break
                else:
                    sample_cur = nusc.get('sample', sample_cur['next'])

            # 坐标变换：global → ego at lcf
            ego_fut_trajs = ego_fut_trajs - np.array(pose_record['translation'])
            rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
            # ego to lidar at lcf
            ego_fut_trajs = ego_fut_trajs - np.array(cs_record['translation'])
            rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
            ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T

            # ============================================================
            # 【类别4：ego_fut_cmd - 驾驶指令，从未来轨迹推断⭐】
            # 注意：这不是真实的导航指令，而是根据未来3秒的横向位移推断的
            # 部署时需要用真实导航系统提供的指令替换！
            # ============================================================
            # drive command according to final fut step offset from lcf
            if ego_fut_trajs[-1][0] >= 2:       # 横向（x）向右偏移 >= 2m
                command = np.array([1, 0, 0])  # Turn Right（右转）
            elif ego_fut_trajs[-1][0] <= -2:   # 横向（x）向左偏移 >= 2m
                command = np.array([0, 1, 0])  # Turn Left（左转）
            else:                               # 横向偏移 < 2m
                command = np.array([0, 0, 1])  # Go Straight（直行）

            # 转换为 offset 格式（每步相对于上一步的位移）
            ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]  # [6, 3] → 6步未来

            # ============================================================
            # 【类别4：ego_lcf_feat - 自车低频控制特征，9维向量】
            # 来源：CAN总线数据 + ego pose 微分计算
            # 9维：[vx, vy, ax, ay, w, length, width, v0, Kappa]
            # ============================================================
            ### ego lcf feat (vx, vy, ax, ay, w, length, width, vel, steer), w: yaw角速度
            ego_lcf_feat = np.zeros(9)

            # 【计算ego速度和角速度 - 从pose微分】
            # 根据odom推算自车速度及加速度
            _, _, ego_yaw = quart_to_rpy(pose_record['rotation'])
            ego_pos = np.array(pose_record['translation'])

            # 获取前后帧的pose（用于微分计算速度）
            if pose_record_prev is not None:
                _, _, ego_yaw_prev = quart_to_rpy(pose_record_prev['rotation'])
                ego_pos_prev = np.array(pose_record_prev['translation'])
            if pose_record_next is not None:
                _, _, ego_yaw_next = quart_to_rpy(pose_record_next['rotation'])
                ego_pos_next = np.array(pose_record_next['translation'])

            assert (pose_record_prev is not None) or (pose_record_next is not None), 'prev token and next token all empty'

            if pose_record_prev is not None:
                # 用前向差分计算
                ego_w = (ego_yaw - ego_yaw_prev) / 0.5  # 角速度 (rad/s), nuScenes 2Hz = 0.5s间隔
                ego_v = np.linalg.norm(ego_pos[:2] - ego_pos_prev[:2]) / 0.5  # 速度标量 (m/s)
                # 分解为横向和纵向速度（考虑yaw角）
                ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi/2), ego_v * math.sin(ego_yaw + np.pi/2)
            else:
                # 用后向差分计算
                ego_w = (ego_yaw_next - ego_yaw) / 0.5
                ego_v = np.linalg.norm(ego_pos_next[:2] - ego_pos[:2]) / 0.5
                ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi/2), ego_v * math.sin(ego_yaw + np.pi/2)

            # 【从CAN总线读取精确的速度和转向数据】
            ref_scene = nusc.get("scene", sample['scene_token'])
            try:
                # 查询CAN总线消息（pose和转向角）
                pose_msgs = nusc_can_bus.get_messages(ref_scene['name'],'pose')
                steer_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'steeranglefeedback')
                pose_uts = [msg['utime'] for msg in pose_msgs]
                steer_uts = [msg['utime'] for msg in steer_msgs]
                ref_utime = sample['timestamp']

                # 找到最近时刻的CAN消息
                pose_index = locate_message(pose_uts, ref_utime)
                pose_data = pose_msgs[pose_index]
                steer_index = locate_message(steer_uts, ref_utime)
                steer_data = steer_msgs[steer_index]

                # initial speed（纵向速度，m/s）
                v0 = pose_data["vel"][0]  # [0] means longitudinal velocity

                # curvature (positive: turn left)（转向曲率）
                steering = steer_data["value"]
                # flip x axis if in left-hand traffic (singapore)
                flip_flag = True if map_location.startswith('singapore') else False
                if flip_flag:
                    steering *= -1
                # 阿克曼转向公式：Kappa = 2*steering_angle / wheelbase
                Kappa = 2 * steering / 2.588  # 2.588是车辆轴距
            except:
                # CAN数据缺失时，用轨迹估算
                delta_x = ego_his_trajs[-1, 0] + ego_fut_trajs[0, 0]
                delta_y = ego_his_trajs[-1, 1] + ego_fut_trajs[0, 1]
                v0 = np.sqrt(delta_x**2 + delta_y**2)  # 估算速度
                Kappa = 0  # 曲率设为0

            # 【组装9维特征向量】
            ego_lcf_feat[:2] = np.array([ego_vx, ego_vy])  # [0:2] 速度 (vx, vy)
            ego_lcf_feat[2:4] = can_bus[7:9]               # [2:4] 加速度 (ax, ay) 从can_bus直接读
            ego_lcf_feat[4] = ego_w                        # [4] 角速度 w
            ego_lcf_feat[5:7] = np.array([ego_length, ego_width])  # [5:7] 车辆尺寸 (长, 宽)
            ego_lcf_feat[7] = v0                           # [7] 纵向速度（CAN）
            ego_lcf_feat[8] = Kappa                        # [8] 转向曲率

            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag
            info['gt_agent_fut_trajs'] = gt_fut_trajs.reshape(-1, fut_ts*2).astype(np.float32)
            info['gt_agent_fut_masks'] = gt_fut_masks.reshape(-1, fut_ts).astype(np.float32)
            info['gt_agent_lcf_feat'] = agent_lcf_feat.astype(np.float32)
            info['gt_agent_fut_yaw'] = gt_fut_yaw.astype(np.float32)
            info['gt_agent_fut_goal'] = gt_fut_goal.astype(np.float32)
            info['gt_ego_his_trajs'] = ego_his_trajs[:, :2].astype(np.float32)
            info['gt_ego_fut_trajs'] = ego_fut_trajs[:, :2].astype(np.float32)
            info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)
            info['gt_ego_fut_cmd'] = command.astype(np.float32)
            info['gt_ego_lcf_feat'] = ego_lcf_feat.astype(np.float32)

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos

def get_global_sensor_pose(rec, nusc, inverse=False):
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])

    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
    if inverse is False:
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
        # translation equivalent writing
        # pose_translation = np.array(sd_cs["translation"])
        # rot_mat = Quaternion(sd_ep['rotation']).rotation_matrix
        # pose_translation = np.dot(rot_mat, pose_translation)
        # # pose_translation = pose[:3, 3]
        # pose_translation = pose_translation + np.array(sd_ep["translation"])
    else:
        sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=False):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: False.
    """
    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmcv.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')
                    [-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec


def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
