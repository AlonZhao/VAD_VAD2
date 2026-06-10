# VAD 训练启动链路全解析

> **定位**：从 `tools/train.py` 命令行出发，完整追踪到 `VAD.forward_train` 算出 loss 的全过程。理解 mmdet "配置驱动 + 注册表 + Runner" 的运行机制。
>
> **回答的核心问题**：配置文件怎么被加载？`type='VAD'` 字符串怎么变成类？训练循环在哪里？loss 怎么算出来的？

---

## 0. 一张总图（先看全局）

```
python tools/train.py VAD_tiny_e2e.py
    │
    ├─[A] Config.fromfile()           解析配置(_base_继承 + exec)
    │                                  此时 type='VAD' 仍是字符串
    │
    ├─[B] importlib.import_module()    加载plugin，装饰器注册所有类
    │                                  type='VAD' ↔ <class VAD> 建立映射
    │
    ├─[C] build_model(cfg.model)       字符串查注册表 → 递归实例化模型树
    │
    ├─[D] build_dataset(cfg.data.train) 同理实例化数据集
    │
    └─[E] custom_train_model()         分发 → custom_train_detector()
              │                          搭台子(dataloader/optimizer/runner/hooks)
              └─ runner.run()  ─────────┐ ★交棒给 mmcv
                                        ↓
                    [mmcv] EpochBasedRunner.run()
                      while epoch < max_epochs:        ← 外层循环
                        for data_batch in data_loader: ← 内层循环
                          model.train_step(data_batch) ← [mmdet] BaseDetector
                            └─ self(**data) = forward()
                                ↓ 回到项目代码
                    [VAD] forward → forward_train → forward_pts_train
                            ├─ pts_bbox_head() 出预测
                            └─ pts_bbox_head.loss() 出loss
                                ↓
                          loss.backward() + optimizer.step()  (在hook里)
```

**三个阶段**：
- **阶段1 准备**（A-D）：把配置变成可运行的模型和数据对象
- **阶段2 循环**（E + mmcv）：epoch/iter 双层循环，在 mmcv 框架里
- **阶段3 计算**（VAD）：每个 iter 回到项目代码算 loss

---

## 1. 阶段1：配置加载与对象构建（tools/train.py）

### 1.1 解析配置（train.py:111）

```python
cfg = Config.fromfile(args.config)   # args.config = "projects/configs/VAD/VAD_tiny_e2e.py"
```

`mmcv.Config.fromfile` 做三件事：
1. **处理 `_base_` 继承**：递归加载基础配置，主文件覆盖
   ```python
   _base_ = ['../datasets/custom_nus-3d.py', '../_base_/default_runtime.py']
   ```
2. **exec 执行**配置文件（本质是 Python 脚本），收集顶层变量
3. 得到 `Config` 对象（类字典，支持 `cfg.model` 点访问）

> **关键**：此时 `cfg.model = dict(type='VAD', ...)` 只是普通字典，`'VAD'` 还只是字符串，类还没加载。

### 1.2 加载 plugin，注册类（train.py:120-144）⭐

```python
if cfg.plugin:                                   # 配置有 plugin=True
    plugin_dir = cfg.plugin_dir                  # 'projects/mmdet3d_plugin/'
    _module_path = 'projects.mmdet3d_plugin'     # 路径转包名
    plg_lib = importlib.import_module(_module_path)  # ← 动态import触发注册
```

`import_module('projects.mmdet3d_plugin')` 执行 `__init__.py` 链：
```
__init__.py: from .VAD import *
  → VAD/__init__.py: from .VAD import VAD
    → 执行 VAD.py 整个文件
      → @DETECTORS.register_module() 装饰器运行
        → DETECTORS 注册表新增 {'VAD': <class VAD>}
```

**装饰器 = 注册机制**：import 一个类文件，文件顶部的 `@xxx.register_module()` 就执行，把"类名→类"存进全局注册表字典。

> 这一步之后，`type='VAD'` 才能在注册表查到类。**bevformer/ 没被 import，所以它的类没进注册表**。

### 1.3 构建模型（train.py:222）

```python
model = build_model(cfg.model, train_cfg=..., test_cfg=...)
```

`build_model` 逻辑：
```python
obj_type = cfg.pop('type')      # 'VAD'
cls = DETECTORS.get(obj_type)   # 查表得 <class VAD>
return cls(**cfg)               # VAD(use_grid_mask=True, ...)
```

**递归构建**：VAD.__init__ → build_head('VADHead') → build_transformer('VADPerceptionTransformer') → build encoder/decoder/coder/assigner/loss... 整棵模型树建起来。

### 1.4 构建数据集（train.py:229）

```python
datasets = [build_dataset(cfg.data.train)]   # type='VADCustomNuScenesDataset' → 查表实例化
```

### 1.5 启动训练（train.py:255）

```python
custom_train_model(model, datasets, cfg, ...)
# 来自 train.py:144 的 import
```

---

## 2. 阶段2：训练循环（apis + mmcv）

### 2.1 custom_train_model 只是分发器（VAD/apis/train.py）

```python
def custom_train_model(model, dataset, cfg, ...):
    if cfg.model.type in ['EncoderDecoder3D']:   # 分割模型(VAD不走)
        assert False
    else:
        custom_train_detector(model, dataset, cfg, ...)  # ← 检测模型走这
```

**不是循环**，只判断模型类型后转发。

### 2.2 custom_train_detector 搭台子（VAD/apis/mmdet_train.py:23-194）

这个函数做**准备工作**，不是循环：

```python
def custom_train_detector(model, dataset, cfg, ...):
    # ① 建数据加载器
    data_loaders = [build_dataloader(ds, ...) for ds in dataset]   # 51行

    # ② 模型上GPU
    model = MMDistributedDataParallel(model.cuda(), ...)            # 70行

    # ③ 建优化器
    optimizer = build_optimizer(model, cfg.optimizer)              # 90行

    # ④ 建 Runner（核心执行器）
    runner = build_runner(cfg.runner, default_args=dict(           # 114行
        model=model, optimizer=optimizer, work_dir=..., logger=...))
    #   cfg.runner = dict(type='EpochBasedRunner', max_epochs=60)

    # ⑤ 注册 hooks（学习率/保存/日志/评测/自定义）
    runner.register_training_hooks(cfg.lr_config, optimizer_config, # 137行
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(eval_hook(...))                            # 174行 验证hook
    for hook_cfg in cfg.custom_hooks:                               # 177行 自定义hook
        runner.register_hook(build_from_cfg(hook_cfg, HOOKS))
        # CustomSetEpochInfoHook

    # ⑥ ★真正开跑★
    runner.run(data_loaders, cfg.workflow)                          # 194行 ← 循环入口
```

**关键就是最后一行 `runner.run()`**，前面全是准备。

### 2.3 循环在 mmcv 的 EpochBasedRunner 里 ★

`runner` 类型是 `EpochBasedRunner`（来自 **mmcv 库**，不在本项目）。`run()` 内部（mmcv源码简化）：

```python
class EpochBasedRunner(BaseRunner):
    def run(self, data_loaders, workflow):
        # workflow=[('train',1)] 表示"训练1个epoch为一轮"
        while self.epoch < self._max_epochs:        # ← 外层循环：EPOCH
            for flow in workflow:
                mode, epochs = flow                  # 'train', 1
                epoch_runner = getattr(self, mode)   # self.train
                for _ in range(epochs):
                    epoch_runner(data_loaders[0])    # 调 self.train()

    def train(self, data_loader):
        self.model.train()
        self.call_hook('before_train_epoch')
        for i, data_batch in enumerate(data_loader): # ← 内层循环：ITER
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True)
            self.call_hook('after_train_iter')       # optimizer.step在这hook里
        self.call_hook('after_train_epoch')

    def run_iter(self, data_batch, train_mode):
        outputs = self.model.train_step(data_batch, self.optimizer)  # ← 调模型
        self.outputs = outputs
```

**两层循环都在这**：
- 外层 `while epoch < max_epochs`
- 内层 `for data_batch in data_loader`

### 2.4 Hook 机制：循环里的插槽

循环里到处是 `call_hook('xxx')`，Hook 是插在固定位置的回调：

```
before_run
  before_train_epoch       ← CustomSetEpochInfoHook 设当前epoch
    before_train_iter
      run_iter (前向+反向)
    after_train_iter        ← OptimizerHook 做 optimizer.step()
                            ← LrUpdaterHook 调学习率
                            ← CheckpointHook 存模型
  after_train_epoch         ← EvalHook 跑验证
after_run
```

配置里的 `lr_config`/`checkpoint_config`/`custom_hooks` 全是往插槽塞回调。

---

## 3. 阶段3：从框架回到项目代码算 loss（VAD.py）

### 3.1 train_step 是继承来的（不在 VAD.py）

```python
class VAD(MVXTwoStageDetector):  # VAD.py:44
    # 没有定义 train_step！
```

`train_step` 继承自 mmdet 的 `BaseDetector`（mmdet源码简化）：
```python
class BaseDetector:
    def train_step(self, data, optimizer):
        losses = self(**data)                    # ← 调 forward()
        loss, log_vars = self._parse_losses(losses)  # 汇总各项loss
        return dict(loss=loss, log_vars=log_vars, num_samples=len(data))
```

`self(**data)` 触发 `__call__` → `forward()`，回到项目代码。

### 3.2 forward 分发（VAD.py:234）

```python
def forward(self, return_loss=True, **kwargs):
    if return_loss:
        return self.forward_train(**kwargs)   # ← 训练走这
    else:
        return self.forward_test(**kwargs)
```

### 3.3 forward_train 主流程（VAD.py:274-403）

三步走：
```python
def forward_train(self, img, img_metas, gt_bboxes_3d, ..., ego_fut_trajs, ...):
    # 步骤1：拆时序——历史帧 vs 当前帧
    len_queue = img.size(1)
    prev_img = img[:, :-1, ...]   # 历史帧
    img = img[:, -1, ...]          # 当前帧

    # 步骤2：历史帧→prev_bev（无梯度，省显存）
    prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

    # 步骤3：当前帧→特征→算loss
    img_feats = self.extract_feat(img=img, img_metas=img_metas)   # ResNet+FPN
    losses = self.forward_pts_train(img_feats, gt_bboxes_3d, ...,  # ← 出loss
                                    prev_bev, ego_fut_trajs=..., ...)
    return losses
```

### 3.4 forward_pts_train 出 loss（VAD.py:166-228）⭐ 终点

干两件事：
```python
def forward_pts_train(self, pts_feats, gt_bboxes_3d, ..., prev_bev, ...):
    # ① 跑 head：BEV编码 → 检测/地图/Motion/Planning 四个head解码
    outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev,
                              ego_his_trajs=..., ego_lcf_feat=...)
    #   outs 含: 检测框/地图/他车轨迹/自车轨迹 等所有预测

    # ② 算 loss：预测outs 和 GT 对比
    loss_inputs = [gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d,
                   map_gt_labels_3d, outs, ego_fut_trajs, ...]
    losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
    return losses
    #   losses 含: det/motion/map/planning(碰撞/边界/方向) 多任务loss
```

**这就是 loss 的诞生地**。`pts_bbox_head` 就是 VADHead，它的 forward 和 loss 方法产出所有预测和损失。

### 3.5 loss 怎么回到优化器

`forward_pts_train` 返回的 losses 字典，一路 return 回 mmcv 的 `run_iter`：
```
forward_pts_train → forward_train → forward → train_step
   → run_iter 拿到 outputs['loss']
   → after_train_iter hook 里：
       OptimizerHook: loss.backward() + optimizer.step() + zero_grad()
```

**梯度更新发生在 hook 里**（OptimizerHook），不在模型代码里——这也是框架/业务分离的体现。

---

## 4. 完整调用栈（一行一跳）

```
tools/train.py:main()
  → Config.fromfile()                          [mmcv] 解析配置
  → importlib.import_module()                  [stdlib] 触发注册
  → build_model()                              [mmdet] 字符串→类→实例化
  → build_dataset()                            [mmdet]
  → custom_train_model()                       [VAD/apis/train.py] 分发
    → custom_train_detector()                  [VAD/apis/mmdet_train.py] 搭台子
      → build_dataloader/optimizer/runner
      → register_hooks()
      → runner.run()                           [mmcv] ★交棒
        → while epoch:                         [mmcv] 外层循环
          → self.train(data_loader)
            → for data_batch:                  [mmcv] 内层循环
              → run_iter()
                → model.train_step()           [mmdet BaseDetector]
                  → self(**data) = forward()   [VAD.py:234] ★回到项目
                    → forward_train()          [VAD.py:274]
                      → obtain_history_bev()    历史帧→prev_bev
                      → extract_feat()          当前帧→img_feats
                      → forward_pts_train()     [VAD.py:166] ★算loss
                        → pts_bbox_head()        VADHead出预测
                        → pts_bbox_head.loss()   出loss
                  ← 返回 losses
            → after_train_iter hook            [mmcv] OptimizerHook
              → loss.backward() + optimizer.step()
```

---

## 5. 三个关键理解（面试常问）

### 5.1 为什么配置写字符串就能用类？
**注册表模式**：`@register_module()` 装饰器在 import 时把"类名→类"存进全局字典，`build_xxx` 用字符串查表实例化。配置与代码解耦，改配置不用改代码。

### 5.2 训练循环为什么在 mmcv 不在项目里？
**框架/业务分离**：epoch/iter 循环、hook 调度、断点续训、日志是所有模型通用的，mmcv 写一次复用。项目只写 `forward_train` 算 loss。

### 5.3 梯度更新在哪？
不在模型代码，在 **mmcv 的 OptimizerHook**（`after_train_iter` 时机）做 `backward()` + `step()`。模型只负责返回 loss 字典。

---

## 6. 类比记忆

```
配置加载   = 点菜（写菜名 type='VAD'）
注册表     = 后厨菜谱墙（装饰器把做法贴上墙）
build_model= 厨师按菜名查菜谱做菜
runner.run = 司仪主持流程（while epoch: for iter:）
hook       = 流程里的固定环节（到点喊人干活）
forward_train = 新人上台表演（算loss）
OptimizerHook = 记分员（backward+step更新）
```

---

## 7. 一句话总结

> `tools/train.py` 先用注册表机制把配置字符串变成模型/数据对象，再交给 **mmcv 的 EpochBasedRunner** 跑 `while epoch: for iter:` 双层循环；每个 iter 通过 `train_step → forward → forward_train → forward_pts_train` 回到 VAD 项目代码算出多任务 loss，最后由 **OptimizerHook** 反向传播更新参数。循环在框架里，loss 在项目里，梯度更新在 hook 里。
