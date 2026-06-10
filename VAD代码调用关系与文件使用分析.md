# VAD 代码调用关系与文件使用分析

> **定位**：梳理项目实际用到哪些文件、哪些是遗留代码，以及核心类的调用链。
>
> **入口**：`projects/configs/VAD/VAD_tiny_e2e.py`（`plugin=True`, `plugin_dir='projects/mmdet3d_plugin/'`）
>
> **机制**：mmdet3d 加载 plugin 时执行顶层 `__init__.py` 完成类注册，配置里的 `type='Xxx'` 字符串从注册表查找类。

---

## 一、核心结论速览

| 类别 | 说明 |
|---|---|
| ⚠️ **最大遗留** | 整个 `bevformer/` 目录（19个.py）完全未被引用 |
| ⚠️ **文件级冗余** | `VAD/modules/transformer.py` 的 `PerceptionTransformer` 未用（配置用的是 `VAD_transformer.py` 的 `VADPerceptionTransformer`） |
| ⚠️ **导出未用** | vovnet/adamw2/position_embedding/embed/visual/lidar_box3d/自定义runner |
| ✅ **真正核心** | VAD.py + VAD_head.py + VAD_transformer.py + modules/(5个) + utils + core/bbox + datasets |

**关键证据**：顶层 `projects/mmdet3d_plugin/__init__.py` 只有 `from .VAD import *`，**没有** `from .bevformer import *`，所以 bevformer 整套根本没被注册。

---

## 二、主调用链（自顶向下）

```
tools/train.py / tools/test.py
    │ from projects.mmdet3d_plugin.VAD.apis.train import custom_train_model
    │ from projects.mmdet3d_plugin.VAD.apis.test import custom_multi_gpu_test
    ↓
配置 VAD_tiny_e2e.py (plugin=True → 加载 mmdet3d_plugin/__init__.py 注册所有类)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ model = dict(type='VAD', ...)                               │
│   → VAD.py: class VAD (@DETECTORS)                          │
│       ├─ img_backbone: ResNet (mmdet标准库)                 │
│       ├─ img_neck: FPN (mmdet标准库)                        │
│       ├─ grid_mask: GridMask (models/utils/grid_mask.py)    │
│       ├─ planner_metric: PlanningMetric                     │
│       │    (VAD/planner/metric_stp3.py)                     │
│       │      └─ get_ade/get_fde                             │
│       │         (core/evaluation/metric_motion.py)          │
│       └─ pts_bbox_head: VADHead                             │
│           → VAD_head.py: class VADHead(DETRHead)            │
│               ├─ transformer: VADPerceptionTransformer      │
│               │    (VAD_transformer.py)                     │
│               │      ├─ encoder: BEVFormerEncoder           │
│               │      │    (modules/encoder.py)              │
│               │      │      ├─ BEVFormerLayer               │
│               │      │      │   (继承 custom_base_          │
│               │      │      │    transformer_layer.py)      │
│               │      │      ├─ TemporalSelfAttention        │
│               │      │      │   (modules/temporal_self_     │
│               │      │      │    attention.py)              │
│               │      │      └─ SpatialCrossAttention +      │
│               │      │         MSDeformableAttention3D      │
│               │      │         (modules/spatial_cross_      │
│               │      │          attention.py)               │
│               │      ├─ decoder: DetectionTransformer       │
│               │      │    Decoder + CustomMSDeformable      │
│               │      │    Attention (modules/decoder.py)    │
│               │      └─ map_decoder: MapDetection           │
│               │         TransformerDecoder                  │
│               │         (VAD_transformer.py)                │
│               │      └─ CustomTransformerDecoder            │
│               │         (VAD_transformer.py, 用于           │
│               │          motion/ego交互)                    │
│               ├─ bbox_coder: CustomNMSFreeCoder             │
│               │    (core/bbox/coders/fut_nms_free_coder.py) │
│               ├─ map_bbox_coder: MapNMSFreeCoder            │
│               │    (core/bbox/coders/map_nms_free_coder.py) │
│               ├─ assigner: HungarianAssigner3D              │
│               │    (core/bbox/assigners/                    │
│               │     hungarian_assigner_3d.py)               │
│               ├─ map_assigner: MapHungarianAssigner3D       │
│               │    (.../map_hungarian_assigner_3d.py)       │
│               │      └─ map_utils.py + match_cost.py        │
│               ├─ traj_lr_warmup (VAD/utils/)                │
│               └─ losses:                                    │
│                    ├─ CD_loss.py (PtsL1Loss/PtsDirCosLoss   │
│                    │   /OrderedPtsL1Loss...)                │
│                    └─ plan_loss.py (PlanCollisionLoss/      │
│                       PlanMapBoundLoss/PlanMapDirectionLoss)│
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ data = dict(type='VADCustomNuScenesDataset', ...)          │
│   → datasets/nuscenes_vad_dataset.py                       │
│       ├─ NuScenesEval_custom                               │
│       │    (datasets/vad_custom_nuscenes_eval.py)          │
│       ├─ CustomNuscenesBox                                  │
│       │    (core/bbox/structures/nuscenes_box.py)          │
│       ├─ [评测时] map_utils/mean_ap.py (eval_map)          │
│       │    → tpfp.py / tpfp_chamfer.py                     │
│       └─ pipeline: datasets/pipelines/                     │
│            ├─ loading.py (LoadMultiViewImageFromFiles)     │
│            ├─ transform_3d.py (Normalize/Pad/PhotoMetric/  │
│            │   RandomScale/ObjectRangeFilter/NameFilter)   │
│            └─ formating.py (CustomDefaultFormatBundle3D/   │
│               CustomCollect3D)                             │
│   → dataloader: datasets/builder.py + samplers/           │
│       (DistributedGroupSampler/DistributedSampler)         │
└─────────────────────────────────────────────────────────────┘
    ↓
runner = EpochBasedRunner (mmcv标准库, 非自定义)
custom_hooks = [CustomSetEpochInfoHook] (VAD/hooks/custom_hooks.py)
optimizer = AdamW (PyTorch标准, 非自定义AdamW2)
```

---

## 三、实际使用的文件清单

### 3.1 VAD 核心（必用）
| 文件 | 角色 |
|---|---|
| `VAD/VAD.py` | 模型主入口 `class VAD` |
| `VAD/VAD_head.py` | 多任务头 `class VADHead` |
| `VAD/VAD_transformer.py` | `VADPerceptionTransformer` + `MapDetectionTransformerDecoder` + `CustomTransformerDecoder` |

### 3.2 VAD/modules（5个用，1个不用）
| 文件 | 状态 | 说明 |
|---|---|---|
| `encoder.py` | ✅ | `BEVFormerEncoder` / `BEVFormerLayer` |
| `spatial_cross_attention.py` | ✅ | `SpatialCrossAttention` / `MSDeformableAttention3D` |
| `temporal_self_attention.py` | ✅ | `TemporalSelfAttention` |
| `decoder.py` | ✅ | `DetectionTransformerDecoder` / `CustomMSDeformableAttention` |
| `custom_base_transformer_layer.py` | ✅ | 被 encoder.py 继承 |
| `multi_scale_deformable_attn_function.py` | ✅ | 被 decoder.py / VAD_transformer.py 引用 |
| `transformer.py` | ❌ | `PerceptionTransformer` 未被引用（冗余） |

### 3.3 utils / planner（全用）
| 文件 | 引用方 |
|---|---|
| `VAD/utils/CD_loss.py` | 配置 loss（PtsL1Loss等） |
| `VAD/utils/plan_loss.py` | 配置 loss（3个规划约束） |
| `VAD/utils/map_utils.py` | map_coder / map_assigner |
| `VAD/utils/traj_lr_warmup.py` | VAD_head.py:22 |
| `VAD/planner/metric_stp3.py` | VAD.py（规划指标） |
| `core/evaluation/metric_motion.py` | metric_stp3.py（get_ade/fde） |

### 3.4 core/bbox（用）
| 文件 | 角色 |
|---|---|
| `coders/fut_nms_free_coder.py` | `CustomNMSFreeCoder` |
| `coders/map_nms_free_coder.py` | `MapNMSFreeCoder` |
| `assigners/hungarian_assigner_3d.py` | `HungarianAssigner3D` |
| `assigners/map_hungarian_assigner_3d.py` | `MapHungarianAssigner3D` |
| `match_costs/match_cost.py` | 各种 Cost |
| `structures/nuscenes_box.py` | `CustomNuscenesBox`（dataset用） |
| `util.py` | normalize_bbox 等工具 |

### 3.5 datasets（用）
| 文件 | 角色 |
|---|---|
| `nuscenes_vad_dataset.py` | `VADCustomNuScenesDataset` |
| `vad_custom_nuscenes_eval.py` | `NuScenesEval_custom`（被dataset引用） |
| `pipelines/loading.py` `transform_3d.py` `formating.py` | 全部transform |
| `builder.py` | build_dataloader（tools引用） |
| `samplers/group_sampler.py` `distributed_sampler.py` | 采样器 |
| `map_utils/mean_ap.py` `tpfp.py` `tpfp_chamfer.py` | 地图评测（条件import，评测时用） |

### 3.6 models / apis / hooks（部分用）
| 文件 | 状态 |
|---|---|
| `models/utils/grid_mask.py` | ✅ VAD.py 用 |
| `models/utils/bricks.py` | ✅ run_time 装饰器，多模块用 |
| `VAD/apis/train.py` `test.py` `mmdet_train.py` | ✅ tools 入口引用 |
| `VAD/hooks/custom_hooks.py` | ✅ `CustomSetEpochInfoHook`（配置用） |
| `core/evaluation/eval_hooks.py` | ✅ `CustomDistEvalHook`（顶层__init__注册，评测hook） |

---

## 四、未使用 / 遗留文件清单

### 4.1 整个 bevformer/ 目录（最大遗留，19个.py）
```
bevformer/
├── __init__.py
├── detectors/         (bevformer.py, bevformer_fp16.py, __init__.py)
├── dense_heads/       (bevformer_head.py, __init__.py)
├── modules/           (7个.py — 和 VAD/modules/ 高度重复，仅差5-7行注释)
├── hooks/             (custom_hooks.py, __init__.py)
├── runner/            (epoch_based_runner.py, __init__.py)
└── apis/              (train.py, test.py, mmdet_train.py, __init__.py)
```
**判据**：顶层 `__init__.py` 不 import bevformer；全项目零交叉引用（唯一匹配是 kitti2waymo.py 一句打印日志的文字）。VAD/modules/ 是它的"复制改名版"。

### 4.2 单文件/单类遗留
| 文件/类 | 判据 |
|---|---|
| `VAD/modules/transformer.py`（PerceptionTransformer） | 配置用 VADPerceptionTransformer，此类无引用 |
| `models/backbones/vovnet.py`（VoVNet） | 导出但配置用 ResNet |
| `models/opt/adamw.py`（AdamW2） | 导出但配置用标准 AdamW |
| `models/utils/position_embedding.py` | 导出无引用 |
| `models/utils/embed.py`（PatchEmbed） | 导出无引用 |
| `models/utils/visual.py`（save_tensor） | 导出无引用 |
| `VAD/runner/epoch_based_runner.py` | 配置用 mmcv 标准 EpochBasedRunner |
| `core/bbox/structures/lidar_box3d.py` | 仅 __init__ 导出，无实际使用 |
| `datasets/nuscenes_eval.py`（老版） | 无任何 import（dataset 用的是 vad_custom_nuscenes_eval.py） |
| `core/evaluation/kitti2waymo.py` | waymo相关，VAD流程无关 |
| `models/hooks/hooks.py` | 存疑，未见配置/代码显式引用 |

---

## 五、为什么有这么多遗留？

VAD 是在 **BEVFormer 代码库基础上改的**，作者保留了原始 bevformer 目录（没删），然后把需要的部分复制到 VAD/ 目录改造。这造成：
- `bevformer/modules/` ↔ `VAD/modules/` 几乎重复（只差注释）
- `bevformer/detectors/bevformer.py` ↔ `VAD/VAD.py`（VAD是改进版）
- 一堆"备选组件"（vovnet/adamw2/各种embed）保留但配置没启用

**实践建议**：
- 读代码只看 `VAD/` + `core/bbox/` + `datasets/` + `models/utils/{grid_mask,bricks}` 即可
- `bevformer/` 整个目录可忽略（甚至可删，不影响 VAD 运行）
- 想精简项目：删 bevformer/、modules/transformer.py、未用的 models 子文件

---

## 六、一句话总结

> VAD 真正运行的代码 = **VAD/（去掉modules/transformer.py）+ core/bbox/ + datasets/ + 少量models工具**；`bevformer/` 整个目录是改造前的遗留母本，与运行链完全无关。
