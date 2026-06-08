# VAD 模型架构详解

> **文档定位**：从代码层面详细解析 VAD 模型的网络结构、各模块功能和数据流
>
> **适用对象**：需要理解模型架构细节、修改模型或面试端到端自动驾驶的工程师

---

## 📑 目录

1. [整体架构概览](#1-整体架构概览)
2. [图像特征提取](#2-图像特征提取)
3. [BEV Encoder](#3-bev-encoder)
4. [Detection Head](#4-detection-head)
5. [Motion Head](#5-motion-head)
6. [Map Head](#6-map-head)
7. [Planning Head](#7-planning-head)
8. [完整的前向传播流程](#8-完整的前向传播流程)

---

## 1. 整体架构概览

### 1.1 顶层结构

```
VAD 模型 (VAD.py)
├── img_backbone (ResNet50/101)      # 图像特征提取
├── img_neck (FPN)                   # 多尺度特征融合
└── pts_bbox_head (VADHead)          # 主要的端到端模块
    ├── BEV Encoder                  # 多视角图像 → BEV 特征
    ├── Detection Head               # 3D 目标检测
    ├── Motion Head                  # 他车轨迹预测
    ├── Map Head                     # 地图元素检测
    └── Planning Head                # 自车轨迹规划
```

### 1.2 关键组件定义

**文件位置**：
- `projects/mmdet3d_plugin/VAD/VAD.py` - 模型主入口（调度器）
- `projects/mmdet3d_plugin/VAD/VAD_head.py` - 核心网络结构
- `projects/mmdet3d_plugin/VAD/VAD_transformer.py` - Transformer 定义
- `projects/mmdet3d_plugin/VAD/modules/` - 自定义模块（encoder/decoder）

### 1.3 数据流概览

```python
# 完整的前向传播流程
输入: 图像 [B, 6, 3, H, W]
  ↓
【阶段1：图像特征提取】
  img_backbone (ResNet50) → [B, 6, C, H', W']
  img_neck (FPN)          → 多尺度特征 list[Tensor]
  ↓
【阶段2：BEV Encoder】
  Spatial Cross-Attention  → 6个相机特征投影到BEV
  Temporal Self-Attention  → 融合上一帧BEV特征
  输出: bev_embed [B, H_bev×W_bev, C] (通常 200×200×256)
  ↓
【阶段3：多任务Head】
  ├─ Detection Head  → 3D框 [N_obj, 7]
  ├─ Motion Head     → 他车轨迹 [N_obj, 6, 6, 2]
  ├─ Map Head        → 地图元素 [N_map, 20, 2]
  └─ Planning Head   → 自车轨迹 [3, 6, 2]
```

---

## 2. 图像特征提取

### 2.1 整体流程

**代码位置**：`VAD.py` 第105-164行

```python
def extract_feat(self, img, img_metas=None, len_queue=None):
    """提取图像特征的完整流程"""
    img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
    return img_feats

def extract_img_feat(self, img, img_metas, len_queue=None):
    """从多视角图像提取2D特征"""
    # 步骤1：img_backbone (ResNet50/101) - 提取语义特征
    img_feats = self.img_backbone(img)
    
    # 步骤2：img_neck (FPN) - 统一通道数、融合多尺度
    if self.with_img_neck:
        img_feats = self.img_neck(img_feats)
    
    return img_feats
```

### 2.2 数据流详解

```
输入图像（每个相机）：1600×900 RGB
    ↓ Resize×0.4 + Pad to 32倍数
预处理后：     640 × 384 × 3
    ↓ 6个相机合并到batch维：(B, 6, 3, 384, 640) → (B×6, 3, 384, 640)
    ↓ 可选 GridMask 数据增强（训练时）
┌──────────────────────────────────────────┐
│ img_backbone: ResNet50（ImageNet预训练）   │
│   共享权重：6个相机用同一个ResNet          │
├──────────────────────────────────────────┤
│ Stage1: 浅层特征（边缘、纹理）             │
│   输出: (B×6, 256, 96, 160)  ← 冻结      │
│ Stage2: 中层特征（部件、形状）             │
│   输出: (B×6, 512, 48, 80)               │
│ Stage3: 深层特征（物体局部）               │
│   输出: (B×6, 1024, 24, 40)              │
│ Stage4: 高层语义（物体类别）← VAD_tiny用这个│
│   输出: (B×6, 2048, 12, 20)              │
└──────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────┐
│ img_neck: FPN（特征金字塔网络）            │
│   压缩通道数：2048 → 256                  │
│   VAD_tiny配置：只输出1个尺度              │
│   输出: (B×6, 256, 12, 20)               │
└──────────────────────────────────────────┘
    ↓ Reshape 回 (B, 6, 256, 12, 20)
img_feats: list[Tensor]  # 长度=1（VAD_tiny）或4（VAD_base）
```

#### **疑问：为什么FPN要把2048维降到256维？**

这看起来很反直觉——ResNet费了半天劲把通道数从3升到2048，FPN又一刀砍到256。但这是个**精心设计的权衡**。

##### **原因1：计算量的巨大差异（最关键）**

后续BEV Encoder要做大量**attention计算**，复杂度是 **O(C²)**：

```python
# BEV query数量 = 10000 (100×100), 6相机×240位置 = 1440
如果保持 C=2048:
  attention计算量 ∝ 10000 × 1440 × 2048² ≈ 60万亿次乘法
降到 C=256:
  attention计算量 ∝ 10000 × 1440 × 256²  ≈ 9400亿次乘法
                                          ↓ 减少64倍！
```

**实际推理速度**：
- C=2048：单帧 ~200ms（无法实时）
- C=256：单帧 ~40ms（25+ FPS）

##### **原因2：显存开销**

```python
# BEV特征显存占用 (B=1, 10000 grids)
C=2048: 10000 × 2048 × 4 bytes = 80 MB（单个BEV）
C=256:  10000 × 256  × 4 bytes = 10 MB（单个BEV）

# 训练时需要保存 当前帧 + 历史3帧 + 中间梯度
C=2048 → ~1 GB     # batch_size只能=1
C=256  → ~128 MB   # batch_size可以到4-8
```

##### **原因3：ResNet的2048维有大量冗余**

ResNet50的2048维是为**ImageNet 1000类分类**设计的，对自动驾驶过剩：

| ImageNet任务 | 自动驾驶任务 | 区别 |
|-------------|------------|------|
| 1000个细粒度类别 | 10个粗粒度类别 | 判别能力需求少 |
| 细分"狗的120个品种" | 只需"车/人/自行车" | 不需要那么细 |
| 单图分类 | 6图融合→BEV | 后续还有大量处理 |

**实证**（来自BEVFormer论文）：C=2048 vs C=256，最终mAP只差**0.5%**，但速度提升**3-4倍**。

##### **原因4：FPN不是单纯降维，是"多尺度融合+降维"**

```python
# VAD_base 用多尺度时的FPN
ResNet输出:
  Stage2: (B×6, 512,  48, 80)  ↘
  Stage3: (B×6, 1024, 24, 40)  → FPN横向连接 + 自顶向下
  Stage4: (B×6, 2048, 12, 20)  ↗
              ↓
FPN输出（统一到256维）:
  P2: (B×6, 256, 48, 80)  ← 融合了浅层细节
  P3: (B×6, 256, 24, 40)  ← 融合了中层特征
  P4: (B×6, 256, 12, 20)  ← 融合了深层语义
  P5: (B×6, 256,  6, 10)  ← 额外下采样
```

FPN做的事：
1. ✅ 把深层语义"下传"到浅层（让浅层也有语义理解）
2. ✅ 把浅层细节"上传"到深层（让深层也有位置精度）
3. ✅ 统一通道数到256（方便后续处理）

**类比**：
- ResNet：每个stage是"专家"（stage4懂语义，stage2懂细节）
- FPN：让所有专家"开会讨论"，统一成256维的"综合报告"

##### **原因5：256是经验最优值（帕累托最优）**

这个数字不是随便定的，是学术界大量实验的结果：

| 维度 | 性能 | 速度 | 显存 | 代表模型 |
|------|------|------|------|---------|
| **128** | 差（-2% mAP） | 快 | 低 | 早期BEVDet |
| **256** | ✅ 好 | ✅ 适中 | ✅ 适中 | BEVFormer / VAD / 主流 |
| **512** | 略好（+0.3% mAP） | 慢2× | 高4× | 追求极致精度 |
| **2048** | 几乎无提升 | 慢8× | 爆显存 | ❌ 无人用 |

**对比其他领域的"经验维度"**：
| 模型 | 维度 | 用途 |
|------|------|------|
| BERT | 768 | NLP（5万词汇量） |
| GPT-2 | 1024 | 通用语言 |
| Vision Transformer | 768 | 通用视觉 |
| **BEVFormer / VAD** | **256** | **自动驾驶（10类）** |

**为什么BEV方案用更少维度？**
- NLP需要更多维度区分5万+词汇
- 自动驾驶类别少（10个），空间信息靠BEV网格表达（不靠通道数）

##### **深入理解：信息瓶颈（Information Bottleneck）**

神经网络学习遵循信息瓶颈原理：

```
输入层（高维、噪音多）
    ↓ 编码（逐步压缩）
隐藏层（低维、提炼信息）  ← 256维在这里
    ↓ 解码（重建任务相关信息）
输出层（任务相关）
```

**关键洞察**：
- 2048维包含的信息**远超**自动驾驶任务需要的
- 压缩到256维，强迫网络**只保留最重要的信息**
- 这反而能**防止过拟合**（去掉ImageNet上的无关特征）

##### **重要事实：通道数 ≠ 信息量**

```
2048维向量（1800维是噪音/冗余）:
  真实信息量 ≈ 248维

256维向量（每个维度都精炼）:
  真实信息量 = 256维

压缩后的256维 > 原始2048维的有效信息
```

**类比**：
- 一本2000页但废话连篇的书（2048维）
- 一本200页但字字珠玑的书（256维）
- **后者信息密度更高**

##### **降维的反向效果实验**

如果不降维会怎样？早期方案尝试过保持高维：

```python
# 方案A：保持 C=2048
BEV Encoder: O(10000 × 1440 × 2048²)
结果：
  - V100单帧推理 ~180ms（无法实时）
  - 显存占用 12 GB（batch_size只能=1）
  - mAP相比 C=256 只高 0.3%

# 方案B：VAD的做法（C=256）
BEV Encoder: O(10000 × 1440 × 256²)
结果：
  - 单帧推理 ~40ms（25 FPS）
  - 显存 2 GB（batch_size可到4）
  - mAP几乎相同
```

**结论**：保持2048维是**得不偿失**的。

##### **类比理解**

**类比1：词典 vs 手册**
```
ResNet Stage4 (2048维):
  像《牛津英语词典》，60万个词条
  包含"自动驾驶"可能用到的所有概念

FPN降到256维:
  像《自动驾驶常用词汇手册》，只留256个最常用词
  "车"、"人"、"路"、"转弯"... 够用了

后续BEV Encoder:
  用这256个词就能"写"出完整的驾驶场景描述
```

**类比2：照片压缩**
```
ResNet输出 (2048维):
  原始RAW照片（100 MB）
  包含大量对"识别物体"无关的信息（噪声、细微色差）

FPN压缩到256维:
  JPEG压缩照片（10 MB）
  去掉无关细节，保留关键信息

实际效果:
  肉眼（下游任务）几乎看不出差别
  但文件小10倍（计算快10倍）
```

##### **小结**

| 维度 | 原因 |
|------|------|
| **计算量** | C=256比C=2048快**64倍**（attention是O(C²)） |
| **显存** | 降低**8倍**，训练时batch size可以更大 |
| **冗余** | ResNet的2048维针对ImageNet 1000类，自动驾驶只需10类 |
| **FPN融合** | 不是单纯降维，而是**多尺度融合+降维** |
| **经验最优** | 256是学术界共识的"性能/效率"最佳点 |
| **防过拟合** | 压缩强迫网络只保留最重要信息 |

**核心思想**：不是"通道越多越好"，而是"**恰到好处**"——256维足够表达自动驾驶场景，再多就是浪费算力。

### 2.3 网络配置（VAD_tiny_e2e）

**配置文件位置**：`projects/configs/VAD/VAD_tiny_e2e.py`

```python
_dim_ = 256          # 统一特征维度
_num_levels_ = 1     # VAD_tiny只用1个尺度（快速版）

img_backbone=dict(
    type='ResNet',
    depth=50,                    # ResNet50
    num_stages=4,                # 4个stage
    out_indices=(3,),            # 只取stage4输出（高层语义）
    frozen_stages=1,             # 冻结stage1（保持底层特征稳定）
    norm_cfg=dict(type='BN', requires_grad=False),
    norm_eval=True,              # BN始终用eval模式
    style='pytorch',
    pretrained='torchvision://resnet50'  # ImageNet预训练权重
)

img_neck=dict(
    type='FPN',
    in_channels=[2048],          # 只接stage4的2048通道
    out_channels=256,            # 压缩到256维
    start_level=0,
    add_extra_convs='on_output', # 在输出上加额外卷积
    num_outs=1,                  # 只输出1个尺度
    relu_before_extra_convs=True
)
```

### 2.4 ResNet 的核心作用

#### **常见误解 vs 实际情况**

| ❌ 常见误解 | ✅ 实际情况 |
|-----------|-----------|
| "ResNet识别出图中有什么物体" | "ResNet把图像编码成特征向量" |
| "ResNet输出'这里有车'" | "ResNet输出2048维特征向量（需Head解读）" |
| "ResNet是分类器" | "ResNet是特征提取器（在VAD中）" |

#### **ResNet 做的事：像素 → 语义特征**

```python
# 输入：原始像素（人眼看得懂，机器看不懂）
img: (384, 640, 3)  # 737,280个RGB数字
# 例如：位置(100, 50) = [245, 240, 235]（只是个白色像素）

# 输出：语义特征（机器能理解）
feat: (12, 20, 2048)  # 491,520个特征值
# 例如：位置(3, 7) = [0.85, 0.02, 0.91, ..., 0.76]
#        ↑      ↑      ↑            ↑
#     "圆形"  "金属"  "深色"      "有玻璃"
#     特征强  特征强  特征强       特征强
#     → 暗示这块区域可能是"轿车"
```

**关键点**：
- 2048个维度**没有人为定义的含义**
- 是网络在ImageNet训练中**自己学出来的抽象特征**
- 可能某个维度对应"圆形"，另一个对应"金属感"，但**没人说得清每个维度具体代表什么**

#### **ResNet 不直接输出"识别结果"**

```
❌ 错误理解：
输入图像 → ResNet → "这里有车、有人、有车道线"

✅ 正确流程：
输入图像 → ResNet → 特征向量（2048维抽象表示）
                      ↓
         [需要后续Head才能解读]
                      ↓
         Detection Head: "这里有车"
         Map Head:       "这里有车道线"  
         Motion Head:    "车要往左开"
```

**类比1：DNA提取**
```
原始细胞    →  ResNet  →    DNA序列      →  解读 → "这是人/狗/猫"
（看不懂）    （提取）    （ATCG，机器可读）   （Head）
```
DNA序列本身**不是"这是人"的结论**，但它**包含了判断这是人的全部信息**。

**类比2：超市条形码**
```
商品  →  扫码枪  →  条形码数字  →  POS系统  →  "可乐 ¥3.5"
        （ResNet）  （特征向量）   （Head）   （结果）
```
条形码只是数字，需要POS系统查询才能得到商品名称。

#### **ResNet 的层级语义**

| Stage | 输出尺寸 | 感受野 | 学到什么 | 例子 |
|-------|---------|-------|---------|------|
| **Stage1** | 96×160×256 | 小 | 低级特征 | 边缘、颜色块、纹理 |
| **Stage2** | 48×80×512 | 中 | 中级特征 | 角点、轮廓、简单形状 |
| **Stage3** | 24×40×1024 | 大 | 高级特征 | 车轮、车窗、人脸等部件 |
| **Stage4** ← VAD用 | 12×20×2048 | 最大 | 语义特征 | "这是一辆车"、"这是车道线" |

**为什么只取Stage4**：VAD后续BEV Encoder需要的是"这个像素代表什么物体"的高层语义，而不是底层纹理。

#### **ResNet 的核心创新：残差连接**

```
传统CNN：x → conv → conv → conv → output
         （层数越多，梯度消失越严重，越深反而越差）

ResNet： x → conv → conv → conv → output
         └─────────────┘  ↑
            残差连接，直接跳过去
         （即使深层学不到东西，至少不会变差）
```
**让网络可以堆到50/101/152层而不退化**，提取更深的语义。

#### **为什么选ResNet50？**

| 备选 | 优势 | 劣势 | VAD的选择 |
|------|------|------|-----------|
| ResNet18 | 快 | 表达能力弱 | ❌ |
| **ResNet50** | **平衡** | **—** | **✅ 默认** |
| ResNet101 | 性能更好 | 慢1.5× | 论文消融实验用 |
| Swin/ViT | SOTA性能 | 显存大、推理慢 | ❌ 不适合实时 |
| VoVNet | 推理更快 | 预训练数据少 | BEVFormer常用 |

**实时部署场景追求"性能/延迟比"**，ResNet50是甜点位置。

#### **ResNet的层级语义与维度变化**

**核心权衡：空间分辨率↓ 换 语义深度↑**

```
输入图像: (384, 640, 3)  ← 原始RGB像素

↓ Stage1: stride=2卷积
Stage1: (B*6, 256, 96, 160)    ← 空间大、通道少、低级特征
        96×160 = 每格覆盖原图 4×4 像素
        256通道 = 基础视觉模式（边缘、颜色、纹理）

↓ Stage2: stride=2卷积
Stage2: (B*6, 512, 48, 80)     ← 空间↓一半、通道↑一倍
        48×80 = 每格覆盖原图 8×8 像素
        512通道 = 中级特征（角点、轮廓、简单形状）

↓ Stage3: stride=2卷积
Stage3: (B*6, 1024, 24, 40)    ← 继续权衡
        24×40 = 每格覆盖原图 16×16 像素
        1024通道 = 高级特征（车轮、车窗等部件）

↓ Stage4: stride=2卷积
Stage4: (B*6, 2048, 12, 20)    ← 空间小、通道多、语义特征
        12×20 = 每格覆盖原图 32×32 像素  ← VAD用这个
        2048通道 = 完整物体语义（轿车、卡车、车道线等）
```

**每过一个stage：空间分辨率 ÷ 2，通道数 × 2**

##### **空间分辨率下降的意义**

```
Stage1 (96×160) - 精细但局部:
  ┌─┬─┬─┬─┬─┬─┬─┬─┐
  │ │ │ │ │ │ │ │ │   每格 4×4 像素
  ├─┼─┼─┼─┼─┼─┼─┼─┤   能分清: 车牌数字、后视镜细节
  │ │ │ │ │ │ │ │ │   但看不出"整体是辆车"
  └─┴─┴─┴─┴─┴─┴─┴─┘

Stage4 (12×20) - 粗糙但整体:
  ┌────┬────┬────┐
  │    │    │    │      每格 32×32 像素
  ├────┼────┼────┤      能分清: "这块是辆车"、"那块是路面"
  │    │    │    │      看不清: 车牌、后视镜等细节
  └────┴────┴────┘
```

**为什么要降低分辨率**：
- ✅ **扩大感受野**：每个位置能"看到"原图更大区域，才能识别完整物体
- ✅ **减少计算量**：12×20的注意力计算比96×160快64倍
- ✅ **降噪**：过滤无关细节（树叶抖动、光影变化）

##### **通道数增加的意义**

每个通道 = 一个**特征检测器**（自动学习，非人工定义）：

```
Stage1 (256通道) - 简单视觉模式:
  通道1: 响应"横向边缘"
  通道2: 响应"纵向边缘"
  通道3: 响应"红色区域"
  通道4: 响应"亮度梯度"
  ...
  通道256: 共256种基础模式

Stage4 (2048通道) - 复杂语义概念:
  通道1:   响应"轿车轮廓"
  通道2:   响应"卡车轮廓"
  通道3:   响应"行人姿态"
  通道4:   响应"车道线"
  通道5:   响应"交通灯"
  通道6:   响应"金属反光"
  ...
  通道2048: 共2048种高级语义
```

**为什么要增加通道数**：
- 浅层只需少量"基础笔画"（横线、竖线、颜色块）
- 深层需表达大量"高级概念"（不同车型、物体类别、运动状态）

##### **核心权衡：空间 vs 语义**

| | 浅层（Stage1） | 深层（Stage4） |
|---|---|---|
| **分辨率** | 96×160（精细） | 12×20（粗糙） |
| **通道数** | 256（概念少） | 2048（概念多） |
| **位置精度** | 像素级 | 区域级 |
| **语义深度** | 模糊（边缘/颜色） | 清晰（物体类别） |
| **感受野** | 小（局部） | 大（整体） |
| **类比** | 放大镜看细节 | 退远看全局 |

##### **数据量守恒定律**

注意每层的**总信息量**大致守恒（实际在压缩）：

```
Stage1: 256  × 96  × 160  = 3,932,160
Stage2: 512  × 48  × 80   = 1,966,080  ↓ 一半
Stage3: 1024 × 24  × 40   =   983,040  ↓ 一半
Stage4: 2048 × 12  × 20   =   491,520  ↓ 一半
```

**CNN = 信息漏斗**：把"无意义的像素噪音"过滤掉，提炼"有意义的语义"。

##### **为什么是 ÷2 / ×2？**

这是CNN的经典设计（ResNet/VGG/EfficientNet都遵循）：

| 操作 | 物理意义 | 实现方式 |
|------|---------|---------|
| 空间 ÷2 | 每次下采样砍半 | stride=2卷积或pooling |
| 通道 ×2 | 用通道翻倍补偿信息损失 | 增加卷积核数量 |

**优势**：
- ✅ 单层计算量保持稳定（H×W×C 大致不变）
- ✅ 内存友好（逐步压缩）
- ✅ 梯度传播稳定

##### **金字塔结构可视化**

```
Stage1: ████████████████████████████  256ch  (96×160)  浅而宽
Stage2: ██████████████  512ch              (48×80)
Stage3: ███████  1024ch                       (24×40)
Stage4: ███  2048ch                              (12×20)  深而窄
        ↑ 通道数（语义维度）             ↑ 空间分辨率（位置精度）
```

**物理意义**：底部"广而浅"（关心局部细节），顶部"窄而深"（关心整体语义）。

##### **VAD为什么只取Stage4？**

```python
img_backbone = ResNet50(out_indices=(3,))  # 只取stage4
```

因为BEV Encoder需要的是**"这个区域代表什么物体"**的语义信息，不需要**"边缘在哪个具体像素"**的细节：

| 任务需求 | 适用Stage |
|---------|----------|
| 检测大物体（车、卡车） | Stage4 ✅（VAD_tiny用） |
| 检测小物体（远处行人） | Stage3+Stage4（VAD_base用多尺度） |
| 像素级分割 | Stage1-4全用（需要细节） |
| 边缘检测 | Stage1（传统视觉） |

VAD_tiny追求速度，只用Stage4；VAD_base用FPN融合多尺度（Stage1-4），小物体检测更准。

##### **类比理解**

**阅读文章的两种方式**：
```
浅层 (高分辨率、低通道) = 逐字母阅读
  优点: 看清每个字母的形状
  缺点: 不知道文章在讲什么

深层 (低分辨率、高通道) = 跳读理解主旨
  优点: 抓住"这是讲自动驾驶的"
  缺点: 记不住具体每个字
```

CNN的设计就是**先逐字母看（浅层），再整段理解（深层）**。

**看地图的两种方式**：
```
浅层 = 拿放大镜看街道
  优点: 每栋房子都看得清
  缺点: 不知道城市整体布局

深层 = 退到高空俯瞰
  优点: 看清城市整体结构
  缺点: 看不清具体某栋房子
```

VAD选择"高空视角"（Stage4），因为规划需要的是"前方有车、有路口"这种宏观信息。

##### **维度变化小结**

| 维度 | 变化方向 | 物理含义 |
|------|---------|---------|
| **空间 (H, W)** | 96×160 → 12×20（↓ 64倍） | 看图变粗糙，但感受野变大 |
| **通道 (C)** | 256 → 2048（↑ 8倍） | 能表达的语义概念变丰富 |
| **总数据量** | 3.9M → 0.5M（↓ 8倍） | 整体在压缩、提炼 |

**核心权衡**：用**空间精度**换**语义深度**——这是所有CNN的灵魂。

### 2.5 关键特性

#### **1. 6个相机共享同一个ResNet50**

```python
# VAD.py 第133-138行
B, N, C, H, W = img.size()           # (B, 6, 3, 384, 640)
img = img.reshape(B * N, C, H, W)    # 把6个相机和batch合并
img_feats = self.img_backbone(img)   # ← 6个相机用同一组权重
```

**为什么共享权重**：
- ✅ 减少参数量（6× → 1×）
- ✅ 增加每个权重的训练数据量（6× 多）
- ✅ 相机参数虽不同，但都是看真实世界的RGB图像，特征提取的"语法"是一样的

#### **2. 只输出2D特征，不知道空间位置**

```python
img_feats[0].shape  # (B, 6, 256, 12, 20)
```
- 12×20是图像下采样后的网格
- 每个网格点的256维向量回答的是"**这个像素看起来像什么**"
- **不知道**这个像素在真实世界的3D位置（那是BEV Encoder的工作）

#### **3. GridMask数据增强（仅训练时）**

```python
# VAD.py 第135-136行
if self.use_grid_mask:
    img = self.grid_mask(img)  # 随机遮挡网格区域
```

强制模型学习**鲁棒特征**——即使部分区域被遮挡（如雨刷、车顶反光），也能正确识别。

参数：`rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7`

#### **4. 冻结策略：frozen_stages=1**

- **Stage1冻结**：保留ImageNet预训练的低层特征（边缘、纹理）
- **Stage2-4微调**：在nuScenes上学习自动驾驶场景的高层语义
- **BN层不更新**：`requires_grad=False, norm_eval=True`

**原因**：低层特征（边缘检测）是通用的，冻结能防止过拟合、加速训练。

#### **5. VAD_tiny vs VAD_base的区别**

| 配置项 | VAD_tiny | VAD_base |
|--------|----------|----------|
| `out_indices` | `(3,)` | `(1,2,3)` |
| `num_outs` | 1 | 4 |
| 输出尺度 | 只有12×20 | 4个尺度（48×80 → 6×10） |
| 速度 | 快 | 慢1.3× |
| 小物体检测 | 可能漏检 | 更准确 |
| 适用场景 | 实时部署 | 离线评测、追求精度 |

### 2.6 ImageNet预训练的价值

ResNet50在ImageNet上预训练（1400万张图像，1000类），已经学会：

| 学到的能力 | 具体表现 |
|-----------|---------|
| 平移不变性 | 车在图像左边/右边都能识别 |
| 光照不变性 | 白天、黄昏、夜晚都能识别 |
| 尺度不变性 | 远处小车、近处大车都能识别 |
| 遮挡鲁棒性 | 部分遮挡也能识别 |
| 通用物体概念 | 车轮、窗户、金属反光等 |

VAD**直接复用这种"看图能力"**，不用从零训练——在nuScenes上只需微调高层特征即可。

### 2.7 下游如何使用这些特征

```python
# 在VAD_head.py中：
img_feats  # (B, 6, 256, 12, 20)
    ↓
BEV Encoder的Spatial Cross-Attention：
    每个BEV query点（俯视图上的一个网格点）
    ↓ 通过相机内外参投影
    投影到6个相机的2D特征图上
    ↓ 在投影位置周围采样特征
    聚合 → 该BEV点的特征
```

**类比**：
- `extract_feat`像是给6张照片各自做了"语义标注"——每个像素打上"车/人/路面/天空"的语义tag
- BEV Encoder再把这些标注**几何投影**到俯视图坐标系，拼成统一的"车顶视角语义图"

### 2.8 ResNet能否替换？

| 替换方案 | 可行性 | 优劣势 |
|---------|-------|--------|
| **不用backbone** | ❌ 不可行 | BEV Encoder要从RGB像素学起，复杂度爆炸 |
| **VGG/Inception** | ✅ 可以 | 但效果不如ResNet50（参数多/性能差） |
| **ResNet101** | ✅ 可以 | 性能略好，慢1.5× |
| **VoVNet-99** | ✅ 可以 | BEVFormer常用，推理更快 |
| **Swin Transformer** | ✅ 可以 | 性能更好，但显存大、推理慢 |

只要符合mmdet的backbone注册规范，改配置即可，不需要改`VAD.py`代码。

### 2.9 时序处理机制

VAD 的一个重要特性是**利用历史帧信息**，但时序融合**不在图像层面**，而是在 **BEV 层面**。

#### **2.9.1 时序输入结构**

```python
# 输入：包含时序的图像序列
img: (B, queue_length, 6, 3, 384, 640)
     ↑      ↑          ↑
   batch  时序帧数    6个相机

# VAD_tiny配置：queue_length = 3
# 意义：3帧历史 + 1帧当前 = 4帧总共
# 时间跨度：3帧 × 0.5s = 1.5秒历史
```

**配置位置**：`projects/configs/VAD/VAD_tiny_e2e.py` 第44行
```python
queue_length = 3  # 历史帧数
```

#### **2.9.2 历史帧与当前帧的拆分**

**代码位置**：`VAD.py` 第376-384行

```python
# 步骤1：拆出历史和当前
len_queue = img.size(1)               # 4
prev_img = img[:, :-1, ...]           # 前3帧 (B, 3, 6, 3, H, W)
img = img[:, -1, ...]                 # 最后1帧 (B, 6, 3, H, W)

# 步骤2：历史帧跑BEV（无梯度）
prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

# 步骤3：当前帧跑完整网络（有梯度）
img_feats = self.extract_feat(img=img, img_metas=img_metas)
```

历史帧和当前帧走**不同分支**：

| | 历史帧分支 | 当前帧分支 |
|---|---------|---------|
| **计算梯度** | ❌ `torch.no_grad()` | ✅ 正常计算梯度 |
| **跑哪些模块** | 只跑到BEV Encoder | 跑完整网络（含4个Head） |
| **输出** | `prev_bev`（BEV特征） | 检测/规划结果 + loss |
| **显存开销** | 低 | 高 |
| **执行模式** | `model.eval()` | `model.train()` |

#### **2.9.3 历史帧逐帧处理**

**代码位置**：`VAD.py` 第246-270行

```python
def obtain_history_bev(self, imgs_queue, img_metas_list):
    """关键：3帧历史串行处理，逐帧累积BEV特征"""
    self.eval()                            # 切到eval（关闭dropout）
    with torch.no_grad():                  # 不算梯度，省显存
        prev_bev = None                    # 初始为空
        
        # 优化：一次性把3帧×6相机过backbone（共18张图）
        bs, len_queue, num_cams, C, H, W = imgs_queue.shape
        imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
        img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
        
        # 逐帧串行处理BEV
        for i in range(len_queue):  # i = 0, 1, 2（对应t-3, t-2, t-1）
            img_metas = [each[i] for each in img_metas_list]
            img_feats = [each_scale[:, i] for each_scale in img_feats_list]
            
            # only_bev=True：只跑BEV Encoder，不跑后续Head
            prev_bev = self.pts_bbox_head(
                img_feats, img_metas, prev_bev,  # ← 上一帧BEV作为输入
                only_bev=True)
        
        self.train()
        return prev_bev  # 最后一帧（t-1）的BEV，喂给当前帧
```

**递归更新过程**：

```
时刻 t-3: img_feat_t-3 → BEV_Encoder(prev_bev=None)      → bev_t-3
时刻 t-2: img_feat_t-2 → BEV_Encoder(prev_bev=bev_t-3)   → bev_t-2
时刻 t-1: img_feat_t-1 → BEV_Encoder(prev_bev=bev_t-2)   → bev_t-1 ← 输出
时刻 t:   img_feat_t   → BEV_Encoder(prev_bev=bev_t-1)   → bev_t
```

每一帧的BEV都"继承"了之前所有帧的信息（类似RNN的隐状态）。最终的`prev_bev`包含了t-3到t-1共1.5秒的历史信息。

#### **2.9.4 关键设计：图像层无时序融合**

```python
# 6相机 × 4帧 = 24张图，但ResNet都是独立处理
img_feats = self.extract_feat(img=imgs_queue)
# 内部: reshape到 (B*4*6, 3, 384, 640) 一起过backbone
# 输出: (B, 4, 6, 256, 12, 20)  ← 4帧6相机的特征是独立的
```

**为什么不在图像层融合？**

| 方案 | 描述 | 问题 |
|------|------|------|
| ❌ **通道拼接** | 把3帧图像在通道维拼接 `(9, H, W)` | 透视投影空间，同一物体在3帧中位置变化巨大 |
| ❌ **3D卷积** | 用3D Conv处理 `(T, H, W, C)` | 计算开销大，仍未解决透视投影对齐问题 |
| ✅ **BEV层融合** | 每帧独立提取2D特征→BEV空间融合 | 俯视图位置变化是简单刚体变换，易对齐 |

- **透视空间**：同一辆车在3帧中可能从图像左侧移到中间，像素位置完全不同
- **BEV空间**：同一辆车只是平移了几个网格，位置变化是刚体变换

#### **2.9.5 用can_bus做空间对齐**

车辆移动后，同一物体在BEV中的位置变了。需要把历史BEV"搬"到当前帧坐标系。

**代码位置**：`VAD_transformer.py` 第230-265行

```python
# 第1步：从can_bus读取ego motion
delta_x = can_bus[0]          # 自车位移x（米）
delta_y = can_bus[1]          # 位移y（米）
ego_angle = can_bus[-2]       # 当前朝向角（弧度）
rotation_angle = can_bus[-1]  # 朝向角变化（弧度）

# 第2步：计算BEV网格的偏移量（单位：格子数）
translation_length = sqrt(delta_x^2 + delta_y^2)
shift_x = translation_length * sin(bev_angle) / grid_length_x / bev_w
shift_y = translation_length * cos(bev_angle) / grid_length_y / bev_h

# 第3步：旋转历史BEV（如果车转弯了）
if self.rotate_prev_bev:
    tmp_prev_bev = rotate(prev_bev, rotation_angle, ...)
```

**物理意义**：

```
时刻 t-1（上一帧坐标系）:        时刻 t（当前帧坐标系，车前进+转向）:
┌──────────────┐                ┌──────────────┐
│   🚗 (ego)   │                │      🚗 (ego)│ ← 自车移动+转向
│   ↑          │                │   ↗          │
│  车A         │                │  车A         │ ← 车A相对位置变了
└──────────────┘                └──────────────┘
   prev_bev      → shift+rotate→  prev_bev_aligned
```

对齐后，`prev_bev`和`bev_t`中**同一物体在同一网格位置**，可以直接做attention。

#### **2.9.6 Temporal Self-Attention融合**

**代码位置**：`modules/encoder.py` 第194-202行

```python
if prev_bev is not None:
    prev_bev = prev_bev.permute(1, 0, 2)
    # 把对齐后的prev_bev和当前bev_query堆在一起
    prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
    hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
        bs*2, len_bev, num_bev_level, 2)

# 当前bev_query通过attention从prev_bev中获取时序信息
# 模型自动学习：哪些历史信息对当前帧有用
```

**学到的时序模式**：
- 车辆运动连续性：上一帧加速 → 这一帧可能继续加速
- 遮挡推理：上一帧某车被遮挡 → 根据轨迹推测位置
- 动态物体过滤：连续几帧不动 → 可能是静态障碍

#### **2.9.7 完整的时序数据流**

```
┌─────────────────────────────────────────────────────────┐
│ 输入: img (B, 4, 6, 3, 384, 640)                        │
└─────────────────────────────────────────────────────────┘
          │
          ├──────────────────────┬──────────────────────┐
          ↓                      ↓                      ↓
   历史帧 t-3            历史帧 t-2            历史帧 t-1
          ↓                      ↓                      ↓
   ResNet+FPN（并行，一次性过backbone）
          ↓                      ↓                      ↓
   img_feat_t-3           img_feat_t-2           img_feat_t-1
          ↓                      ↓                      ↓
┌─────────────────────────────────────────────────────────┐
│ 串行递归（no_grad）:                                     │
│   prev_bev = BEV_Encoder(img_feat_t-3, None)           │
│   prev_bev = BEV_Encoder(img_feat_t-2, prev_bev)       │
│   prev_bev = BEV_Encoder(img_feat_t-1, prev_bev)       │ ← 输出
└─────────────────────────────────────────────────────────┘
                          ↓ prev_bev (t-1时刻的BEV)
┌─────────────────────────────────────────────────────────┐
│ 当前帧 (t)                                               │
│   img_feat_t ← ResNet+FPN (有梯度)                      │
│        ↓                                                 │
│   BEV_Encoder(img_feat_t, prev_bev, can_bus_t):        │
│     ├─ Spatial Cross-Attention: 6相机投影到BEV          │
│     └─ Temporal Self-Attention: 融合prev_bev（已对齐）  │
│        ↓                                                 │
│   bev_embed_t (B, 10000, 256)                           │
│        ↓                                                 │
│   4个Head（Detection / Motion / Map / Planning）        │
└─────────────────────────────────────────────────────────┘
```

#### **2.9.8 训练 vs 推理的差异**

##### **训练模式：每个batch独立，重新计算历史**

```python
for batch in dataloader:
    img (B, 4, 6, 3, H, W)
    prev_bev = obtain_history_bev(img[:, :3])  # 3帧串行
    当前帧前向 (用prev_bev)
    计算loss
```

- ✅ 每个batch看到完整的1.5秒历史上下文
- ✅ 训练稳定（不依赖跨batch状态）
- ❌ 计算开销大：(3帧+1帧)×6相机 = 24次特征提取

##### **推理模式：跨帧缓存，仅前向当前帧**

**代码位置**：`VAD.py` 第93-101行，第433-472行

```python
self.prev_frame_info = {
    'prev_bev': None,        # 上一帧的BEV特征
    'scene_token': None,     # 当前场景标识
    'prev_pos': 0,
    'prev_angle': 0
}

def forward_test(self, img_metas, img, ...):
    # 检查是否换场景
    if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
        self.prev_frame_info['prev_bev'] = None  # 清空历史
    
    # 计算ego motion（相对于上一帧）
    if self.prev_frame_info['prev_bev'] is not None:
        img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
        img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
    
    # 单帧推理
    new_prev_bev, bbox_results = self.simple_test(
        img=img[0],
        prev_bev=self.prev_frame_info['prev_bev'],  # ← 用缓存的BEV
        ...)
    
    # 缓存给下一帧
    self.prev_frame_info['prev_bev'] = new_prev_bev
```

| 对比维度 | 训练 | 推理 |
|---------|------|------|
| **历史帧处理** | 每batch重新计算 | 跨帧缓存 |
| **计算量** | 4帧×6相机 = 24张图 | 1帧×6相机 = 6张图 |
| **梯度** | 当前帧有，历史帧无 | 全无 |
| **场景切换** | batch间独立 | 检测token变化，清空缓存 |
| **显存** | 较高 | 低 |
| **典型FPS** | — | 25+ FPS |

#### **2.9.9 场景切换处理**

**代码位置**：`VAD.py` 第433-437行

```python
if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
    self.prev_frame_info['prev_bev'] = None  # 清空历史BEV
self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']
```

**为什么要清空？**

nuScenes中每个`scene`是一段连续的30秒驾驶片段。不同scene之间：
- ❌ ego位姿不连续（可能跳到不同地点）
- ❌ can_bus的delta失效
- ❌ 上一个scene的BEV特征对当前没有任何帮助

不清空会出问题：
```
Scene A最后一帧: ego在波士顿市中心
                ↓ 不清空
Scene B第一帧:   ego在新加坡机场 ← prev_bev包含波士顿信息！
```

#### **2.9.10 时序处理的两个关键假设**

##### **假设1：自车运动可知（can_bus提供）**

```python
# 直接从CAN总线读取高精度运动信息
delta_x, delta_y = can_bus[0], can_bus[1]
rotation_angle = can_bus[-1]
```

- ✅ 精度高（车辆IMU/轮速编码器）
- ✅ 无延迟
- ❌ 依赖车端硬件（无CAN接口的车无法使用）

##### **假设2：他车运动未知（模型学习推理）**

`prev_bev`中的他车信息可能"过时"。解决方案：
- Temporal Self-Attention让模型自己学：
  - "这辆车上一帧速度vx → 本帧应该在B位置"
  - "这个行人静止 → 本帧仍在原位"
- **当前帧的Spatial Cross-Attention才是ground truth**（实时观测）
- `prev_bev`只提供**时序提示**和**运动连续性先验**

#### **2.9.11 时序处理小结**

| # | 设计 | 目的 | 代码位置 |
|---|------|------|---------|
| **1** | 图像层独立处理 | 透视空间难对齐，留到BEV层 | `VAD.py:138-144` |
| **2** | 历史帧串行递归 | 逐帧累积时序信息（类似RNN） | `VAD.py:246-270` |
| **3** | can_bus空间对齐 | 把历史BEV变换到当前坐标系 | `VAD_transformer.py:230-265` |
| **4** | Temporal Self-Attention融合 | 学习哪些历史信息有用 | `modules/encoder.py:194-202` |
| **5** | 推理时跨帧缓存 | 避免重复计算，提升实时性 | `VAD.py:433-472` |
| **6** | 场景切换清空 | 避免跨场景信息污染 | `VAD.py:433-437` |

**核心优势**：
- ✅ **效率**：推理时只处理当前帧，历史信息靠缓存的BEV
- ✅ **鲁棒**：BEV空间对齐比图像空间简单（刚体变换 vs 透视变形）
- ✅ **灵活**：可选时序（`video_test_mode=False`时退化为单帧）

**与其他方案对比**：

| 方案 | 时序融合位置 | 对齐方式 | 代表 |
|------|------------|---------|------|
| **VAD** | BEV层 | can_bus几何变换 | 本文 |
| **BEVFormer** | BEV层 | Deformable Attention学习偏移 | ECCV 2022 |
| **FIERY** | BEV层 | 3D Conv + GRU | ICCV 2021 |
| **传统方案** | 图像层 | Optical Flow | LSS等 |

VAD的设计在**效率**和**准确性**之间取得了较好平衡。

---

## 3. VAD Head 主体

### 3.0 VADHead 结构总览

**代码位置**：`VAD_head.py` 第72-433行（类定义与初始化）

VADHead 继承自 mmdet 的 `DETRHead`，是 VAD 模型的"主网络"，包含 **5 大功能模块**、约 **30 个子模块**。

#### **3.0.1 整体架构图**

```
VADHead (继承 DETRHead)
│
├─ 【一】BEV 编码层 → 把图像特征转成 BEV
│   └─ self.transformer (VADPerceptionTransformer)
│       ├─ encoder        : BEVFormerEncoder × 3层
│       ├─ decoder        : DetectionTransformerDecoder × 3层
│       ├─ map_decoder    : MapDetectionTransformerDecoder × 3层
│       ├─ level_embeds   : FPN多尺度位置编码
│       ├─ cams_embeds    : 6个相机的位置编码
│       ├─ can_bus_mlp    : ego运动信息编码
│       └─ reference_points / map_reference_points
│
├─ 【二】Query 集合 → 5 类可学习的"槽位"
│   ├─ self.bev_embedding         : 100×100 = 10000 个 BEV 网格
│   ├─ self.query_embedding       : 300 个 detection 槽位
│   ├─ self.map_query_embedding   : 100×20 = 2000 个 map 槽位
│   ├─ self.motion_mode_query     : 6 个多模态行为
│   └─ self.ego_query             : 1 个自车槽位
│
├─ 【三】预测分支（FFN Heads）→ 把 query 解码成具体预测
│   ├─ self.cls_branches × 3       : 检测分类（10类）
│   ├─ self.reg_branches × 3       : 检测回归（10维box）
│   ├─ self.traj_branches × 1      : 他车轨迹预测
│   ├─ self.traj_cls_branches × 1  : 他车轨迹分类
│   ├─ self.map_cls_branches × 3   : 地图分类（3类）
│   ├─ self.map_reg_branches × 3   : 地图回归（box+pts）
│   └─ self.ego_fut_decoder        : 自车规划轨迹（3模态×6步×2）
│
├─ 【四】交互模块 → Motion / Planning 的注意力
│   ├─ self.lane_encoder         : LaneNet（车道线 polyline 编码）
│   ├─ self.motion_decoder       : agent ↔ agent 自交互（1层）
│   ├─ self.motion_map_decoder   : agent ↔ map 跨交互（1层）
│   ├─ self.ego_agent_decoder    : ego ↔ agent 跨交互（1层）
│   ├─ self.ego_map_decoder      : ego ↔ map 跨交互（1层）
│   ├─ self.agent_fus_mlp        : 多模态特征融合
│   └─ self.pos_mlp / pos_mlp_sa : 位置编码 MLP
│
└─ 【五】辅助模块
    ├─ self.bbox_coder        : 检测框编解码（normalize/denormalize）
    ├─ self.map_bbox_coder    : 地图框编解码
    ├─ self.map_assigner      : 地图的匈牙利匹配器
    ├─ self.map_sampler       : 地图采样器（PseudoSampler）
    └─ self.loss_*            : 13+ 个损失函数对象
```

#### **3.0.2 【一】BEV 编码层：self.transformer**

VADHead 最重要的子模块，内部串联 3 个 Transformer。

**代码位置**：`VAD_transformer.py` 第147-183行

```python
self.transformer = VADPerceptionTransformer(
    encoder = BEVFormerEncoder(num_layers=3),
    decoder = DetectionTransformerDecoder(num_layers=3),
    map_decoder = MapDetectionTransformerDecoder(num_layers=3),
    embed_dims = 256,
    num_cams = 6,
    rotate_prev_bev = True,   # 时序对齐时旋转prev_bev
    use_can_bus = True,       # 用CAN总线信息
)
```

**3 个内部 Transformer 的角色**：

| 模块 | 层数 | 输入 | 输出 | 作用 |
|------|------|------|------|------|
| `encoder` | 3 | 6相机特征 + bev_queries + prev_bev | `bev_embed (B, 10000, 256)` | 多视角→BEV |
| `decoder` | 3 | bev_embed + 300个object_query | `hs (3, B, 300, 256)` | BEV→检测特征 |
| `map_decoder` | 3 | bev_embed + 2000个map_query | `map_hs (3, B, 2000, 256)` | BEV→地图特征 |

**重要**：每个 decoder 是 **3 层**（VAD_tiny 配置），深监督输出每层结果 → `hs.shape[0] = 3`。

**transformer 辅助模块**：

```python
# 多尺度/多相机位置编码
self.level_embeds = nn.Parameter(...)  # FPN 4个尺度的编码
self.cams_embeds = nn.Parameter(...)   # 6个相机的编码

# CAN总线信息编码（ego motion）
self.can_bus_mlp = nn.Sequential(
    nn.Linear(18, 256), nn.ReLU(), 
    nn.Linear(256, 256), nn.LayerNorm(256))

# 参考点生成
self.reference_points = nn.Linear(256, 3)      # 检测：3D点(x,y,z)
self.map_reference_points = nn.Linear(256, 2)  # 地图：2D点(x,y)
```

#### **3.0.3 【二】Query 集合：5 类"槽位"**

VAD 用 5 类 `nn.Embedding` 作为可学习的 Query（DETR范式）。

**代码位置**：`VAD_head.py` 第375-389行

```python
# 1. BEV 网格（最稠密）
self.bev_embedding = nn.Embedding(100*100, 256)  # 10000个256维

# 2. 检测槽位
self.query_embedding = nn.Embedding(300, 512)    # 300个512维（含位置）

# 3. 地图槽位（instance_pts 模式）
self.map_instance_embedding = nn.Embedding(100, 512)     # 100条线
self.map_pts_embedding = nn.Embedding(20, 512)           # 每条20个点
# 组合：(100, 1, 512) + (1, 20, 512) → (100, 20, 512) → flatten到2000

# 4. 多模态行为
self.motion_mode_query = nn.Embedding(6, 256)    # 6种行为模式

# 5. 自车
self.ego_query = nn.Embedding(1, 256)
```

**Query 对照表**：

| Query | 数量 | 维度 | 物理含义 |
|-------|------|------|---------|
| bev_embedding | 10000 | 256 | 100×100 俯视图网格点 |
| query_embedding | 300 | 512 | 物体检测的 300 个候选槽位 |
| map_query_embedding | 2000 | 512 | 100 条线 × 20 个点的查询 |
| motion_mode_query | 6 | 256 | 6 种行为（直/左/右/加/减/...） |
| ego_query | 1 | 256 | 自车的查询向量 |

**为什么检测/地图 query 是 512 维？**  
嵌入了**位置 + 内容**两部分（256+256=512），forward 时会拆分：
```python
# forward 中的用法
object_query_embeds = self.query_embedding.weight  # (300, 512)
# 拆成 content(256) + pos(256)，分别用于 query 和 query_pos
```

#### **3.0.4 【三】预测分支：FFN Heads**

每个 query 经 transformer 处理后，用 **MLP** 输出具体预测。结构统一：`Linear → ReLU/LayerNorm → Linear`。

**代码位置**：`VAD_head.py` 第286-331行（定义模板）

```python
# Detection 分类分支（模板）
cls_branch = Sequential(
    Linear(256, 256), LayerNorm(256), ReLU,
    Linear(256, 10)   # 10类：car/truck/.../traffic_cone
)

# Detection 回归分支
reg_branch = Sequential(
    Linear(256, 256), ReLU,
    Linear(256, 10)   # 10维：[x,y,w,l,z,h,sin(yaw),cos(yaw),vx,vy]
)

# Motion 轨迹分支（输入512维 = 多模态拼接后）
traj_branch = Sequential(
    Linear(512, 512), ReLU,
    Linear(512, 12)   # 6步×2维xy = 12
)

# Map 点序列分支
map_reg_branch = Sequential(
    Linear(256, 256), ReLU,
    Linear(256, 10)   # box(4) + pts(20×2编码后10维)
)

# Planning 自车分支（输入514维 = 2×256 + 2个CAN特征）
ego_fut_decoder = Sequential(
    Linear(514, 514), ReLU,
    Linear(514, 36)   # 3模态×6步×2 = 36
)
```

**克隆为每层 decoder 的分支**（第354-372行）：

```python
# 深监督：3层decoder各自有独立的分支
self.cls_branches = _get_clones(cls_branch, 3)  # 3个独立cls_branch
self.reg_branches = _get_clones(reg_branch, 3)
self.map_cls_branches = _get_clones(map_cls_branch, 3)
self.map_reg_branches = _get_clones(map_reg_branch, 3)

# Motion/Planning 只有1层（接在最后）
self.traj_branches = _get_clones(traj_branch, 1)
self.ego_fut_decoder = ego_fut_decoder  # Sequential，不克隆
```

**分支清单**：

| 分支 | 数量 | 输入维度 | 输出维度 | 含义 |
|------|------|---------|---------|------|
| `cls_branches` | 3 | 256 | 10 | 检测分类（10类） |
| `reg_branches` | 3 | 256 | 10 | 检测回归 |
| `traj_branches` | 1 | 512 | 12 | 他车 6 步轨迹 |
| `traj_cls_branches` | 1 | 512 | 1 | 他车模态置信度 |
| `map_cls_branches` | 3 | 256 | 4 | 地图分类（3类+背景） |
| `map_reg_branches` | 3 | 256 | 10 | 地图回归 |
| `ego_fut_decoder` | 1 | 514 | 36 | 自车 3 模态轨迹 |

#### **3.0.5 【四】交互模块：Motion / Planning 的级联**

**代码位置**：`VAD_head.py` 第388-430行

```python
# 4 个 Transformer Decoder（每个 1 层）
self.motion_decoder = build_transformer_layer_sequence(
    dict(type='CustomTransformerDecoder', num_layers=1, ...))
self.motion_map_decoder = build_transformer_layer_sequence(...)
self.ego_agent_decoder = build_transformer_layer_sequence(...)
self.ego_map_decoder = build_transformer_layer_sequence(...)

# 辅助模块
self.lane_encoder = LaneNet(256, 128, 3)  # polyline → 单个特征向量
self.agent_fus_mlp = Sequential(          # 多模态融合
    Linear(6*2*256, 256), LayerNorm(256), ReLU, Linear(256, 256))
self.pos_mlp = Linear(2, 256)             # 2D位置 → 256维编码
self.ego_agent_pos_mlp = Linear(2, 256)
self.ego_map_pos_mlp = Linear(2, 256)
```

**4 次交互的级联**（forward中的调用顺序）：

```
        Detection 输出 (300 agents)
              │
              ↓ + motion_mode_query (6)
       ┌─ motion_decoder ─┐  ← agent 之间 self-attention
       │   (1层)          │
       └────────┬─────────┘
                ↓
       ┌ motion_map_decoder ┐  ← agent ↔ map cross-attention
       │   (1层)            │    用 Map Head 的预测 lane_preds
       └────────┬───────────┘
                ↓
            agent 特征（含多模态轨迹）
                │
                ↓ 选高置信度 agent
       ┌ ego_agent_decoder ┐  ← ego ↔ agent cross-attention
       │   (1层)           │
       └────────┬──────────┘
                ↓
       ┌  ego_map_decoder  ┐  ← ego ↔ map cross-attention
       │   (1层)           │    用 Map Head 的预测
       └────────┬──────────┘
                ↓ + ego_lcf_feat (CAN)
          ego_fut_decoder (MLP)
                ↓
         3 模态规划轨迹
```

**关键依赖链**：
```
Detection → Motion → Planning
Map       ────────→ Planning
```

Planning 用的是**预测的 Motion 和 Map**，不是 GT（端到端设计）。

#### **3.0.6 【五】辅助模块**

**编解码器**（坐标归一化 ↔ 真实坐标）：

```python
self.bbox_coder = NMSFreeCoder(
    pc_range=[-15, -30, -2, 15, 30, 2], ...)
self.map_bbox_coder = NMSFreeCoder(...)
```

**匈牙利匹配器**（DETR 训练必需）：

```python
self.map_assigner = MapHungarianAssigner3D(
    cls_cost=dict(type='FocalLossCost', weight=2.0),
    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
    pts_cost=dict(type='ChamferCost', weight=5.0), ...)
self.map_sampler = PseudoSampler()  # DETR不做实际采样
```

**13+ 个损失对象**（在 3.4 章已详细分析）：

```python
self.loss_traj, self.loss_traj_cls
self.loss_map_cls, self.loss_map_bbox, self.loss_map_iou,
self.loss_map_pts, self.loss_map_dir
self.loss_plan_reg, self.loss_plan_bound,
self.loss_plan_col, self.loss_plan_dir
# Detection 的 loss_cls/loss_bbox 继承自父类 DETRHead
```

#### **3.0.7 VADHead 的"输入-处理-输出"全景**

```
┌──────────────────────────────────────────────────────┐
│ 输入                                                  │
│   mlvl_feats: (B, 6, 256, 12, 20)   ← FPN特征        │
│   prev_bev:   (B, 10000, 256)        ← 历史BEV        │
│   ego_his_trajs, ego_lcf_feat       ← 自车信息       │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ 【一】BEV Encoder (transformer.encoder × 3层)         │
│   输入 + bev_embedding(10000)                         │
│   → bev_embed (B, 10000, 256)                        │
└──────────────────────────────────────────────────────┘
           ↓ bev_embed           ↓ bev_embed
┌──────────────────┐   ┌─────────────────────┐
│ 【一】Detection   │   │ 【一】Map Decoder    │
│   Decoder × 3层   │   │   × 3层              │
│   → hs            │   │   → map_hs           │
└────────┬──────────┘   └──────────┬──────────┘
         ↓                          ↓
┌────────────────┐       ┌──────────────────┐
│【三】cls × 3    │       │【三】map_cls × 3  │
│【三】reg × 3    │       │【三】map_reg × 3  │
└────────┬───────┘       └──────────┬───────┘
         ↓                          ↓
         └─────────┐     ┌──────────┘
                   ↓     ↓
        ┌──────────────────────────┐
        │ 【四】motion_decoder      │
        │      + motion_mode_query  │
        │      ↓                    │
        │ 【四】motion_map_decoder   │
        │      ↓                    │
        │ 【三】traj_branches       │
        │      → 他车多模态轨迹      │
        └──────────┬───────────────┘
                   ↓ + ego_query
        ┌──────────────────────────┐
        │ 【四】ego_agent_decoder    │
        │      ↓                    │
        │ 【四】ego_map_decoder      │
        │      ↓ + ego_lcf_feat     │
        │ 【三】ego_fut_decoder     │
        │      → 自车规划轨迹        │
        └──────────────────────────┘
                   ↓
              outs (dict)
```

#### **3.0.8 关键观察**

**1. 模块复用度高**

| 模块 | 复用次数 | 用法 |
|------|---------|------|
| `Transformer Decoder` | 5 处 | detection / map / motion / ego-agent / ego-map |
| `LaneNet` | 2 处 | lane_encoder, ego_his_encoder |
| `MLP (Linear+ReLU)` | 7+ 处 | 各种位置编码、特征融合 |

VAD 设计高度模块化——同一种"积木"在不同位置使用。

**2. 参数量分布（粗估）**

| 模块 | 参数量占比 |
|------|----------|
| `transformer.encoder` (BEV) | ~40% |
| `transformer.decoder` × 3 + `map_decoder` × 3 | ~30% |
| 4 个交互 decoder | ~15% |
| 各种 branches (FFN) | ~10% |
| Query embeddings | ~5% |

BEV Encoder 是最大头——处理 6 相机 × 多尺度的 attention。

**3. VAD_tiny vs VAD_base 对比**

| 配置 | tiny | base |
|------|------|------|
| `bev_h × bev_w` | 100 × 100 | 200 × 200 |
| `_dim_` | 256 | 256 |
| `_num_levels_` (FPN尺度) | 1 | 4 |
| 各 decoder `num_layers` | 3 | 6 |

主要差异在 BEV 分辨率、FPN 尺度数、decoder 层数——结构相同。

### 3.1 VADHead.forward 完整流程

**代码位置**：`VAD_head.py` 第481-810行

**函数签名**：
```python
def forward(self,
            mlvl_feats,      # FPN特征 [(B, 6, 256, 12, 20)]
            img_metas,       # 相机内外参、can_bus等
            prev_bev=None,   # 历史BEV特征 (B, 10000, 256)
            only_bev=False,  # 是否只跑BEV Encoder
            ego_his_trajs=None,  # 自车历史轨迹
            ego_lcf_feat=None,   # CAN总线特征
):
```

**调用关系**：
```python
# VAD.py:220
outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev,
                          ego_his_trajs=..., ego_lcf_feat=...)
        ↓ Python的__call__机制
# VAD_head.py:481
VADHead.forward(self, mlvl_feats=pts_feats, ...)
```

#### **3.1.1 整体数据流**

```
┌──────────────────────────────────────────────────────────┐
│ 输入                                                      │
│   mlvl_feats:    [(B, 6, 256, 12, 20)]  ← FPN特征        │
│   prev_bev:      (B, 10000, 256)         ← 历史BEV        │
│   ego_his_trajs: (B, 1, 2, 3)            ← 自车历史       │
│   ego_lcf_feat:  (B, 1, 9)               ← CAN总线        │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ 阶段1: 准备Query (第505-520行)                           │
│   bev_queries:        (10000, 256)   ← BEV网格           │
│   object_query_embeds:(300, 512)     ← 检测query         │
│   map_query_embeds:   (2000, 512)    ← 地图query         │
│   bev_pos:            (10000, 256)   ← BEV位置编码       │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ 阶段2: Transformer主干 (第535-554行)                      │
│   if only_bev:                                           │
│     return transformer.get_bev_features(...)  ← 历史帧用  │
│   else:                                                  │
│     outputs = transformer(...)  ← 完整流程                │
│   返回: bev_embed, hs, map_hs, ...                       │
└──────────────────────────────────────────────────────────┘
                          ↓
        ┌───────────────┴────────────────┬───────────────┐
        ↓                                ↓               ↓
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 阶段3:       │   │ 阶段4:       │   │ 阶段5:       │
│ Detection    │   │ Map          │   │ Motion       │
│ 解码         │   │ 解码         │   │ Head         │
│ (569-595)    │   │ (597-615)    │   │ (617-697)    │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       ↓                  ↓                  ↓
  检测框预测         地图元素预测      多模态轨迹预测
                          │                  │
                          └────────┬─────────┘
                                   ↓
                    ┌──────────────────────────┐
                    │ 阶段6: Planning Head     │
                    │ (709-794)                │
                    │  ego ↔ agent 交互        │
                    │  ego ↔ map 交互          │
                    │  + CAN特征拼接           │
                    └──────────────────────────┘
                                   ↓
                         自车3模态轨迹预测
```

#### **3.1.2 阶段1：准备Query（第505-520行）**

VAD使用**DETR范式**：用可学习的Query表示"槽位"。

```python
# 提取4类Query
object_query_embeds = self.query_embedding.weight       # (300, 512)
map_query_embeds = self.map_query_embedding.weight      # (2000, 512)
bev_queries = self.bev_embedding.weight                 # (10000, 256)
bev_pos = self.positional_encoding(bev_mask)            # (10000, 256)
```

**4类Query对比**：

| Query类型 | 数量 | 维度 | 用途 | 物理含义 |
|----------|------|------|------|---------|
| `bev_queries` | 10000 | 256 | BEV网格点 | 100×100俯视图网格 |
| `object_query_embeds` | 300 | 512 | 物体检测 | 300个物体"槽位" |
| `map_query_embeds` | 2000 | 512 | 地图元素 | 100条线×20点=2000个槽位 |
| `bev_pos` | 10000 | 256 | 位置编码 | 正弦位置编码 |

**Query的本质**：
- 可学习参数（`nn.Embedding`），训练中学到"应该关注哪些位置"
- DETR范式：用N个query表示N个潜在的预测对象
- 避免手工设计anchor（传统检测的做法）

#### **3.1.3 阶段2：Transformer主干（第522-554行）**

```python
if only_bev:  # 历史帧分支
    return self.transformer.get_bev_features(
        mlvl_feats, bev_queries, ..., prev_bev)
else:         # 当前帧分支（完整流程）
    outputs = self.transformer(
        mlvl_feats, bev_queries, object_query_embeds,
        map_query_embeds, ..., prev_bev)
```

**两种执行路径**：

| 场景 | `only_bev` | 调用函数 | 返回 |
|------|-----------|---------|------|
| 历史帧处理 | `True` | `get_bev_features` | bev_embed (Tensor) |
| 当前帧推理 | `False` | `transformer(...)` | outputs (7个Tensor的tuple) |

**`self.transformer`内部串联**（VADPerceptionTransformer）：
1. **BEV Encoder**：Spatial Cross-Attention + Temporal Self-Attention
2. **Detection Decoder**：6层Transformer Decoder
3. **Map Decoder**：6层Transformer Decoder

**outputs解包（第553-554行）**：
```python
bev_embed, hs, init_reference, inter_references, \
    map_hs, map_init_reference, map_inter_references = outputs
```

| 变量 | 形状 | 含义 |
|------|------|------|
| `bev_embed` | `(B, 10000, 256)` | BEV特征图 |
| `hs` | `(6, B, 300, 256)` | 检测query经6层decoder的特征（深监督） |
| `init_reference` | `(B, 300, 3)` | 检测query初始参考点（3D xyz） |
| `inter_references` | `(6, B, 300, 3)` | 每层decoder后的更新位置 |
| `map_hs` | `(6, B, 2000, 256)` | 地图query经6层decoder的特征 |
| `map_init_reference` | `(B, 2000, 2)` | 地图query初始参考点（2D xy） |
| `map_inter_references` | `(6, B, 2000, 2)` | 每层decoder后的更新位置 |

#### **3.1.4 阶段3：Detection解码（第569-595行）**

**深监督机制**：6层decoder各自输出预测，都参与loss计算。

```python
for lvl in range(6):  # 6层decoder
    if lvl == 0:
        reference = init_reference
    else:
        reference = inter_references[lvl - 1]
    
    # 分类分支
    outputs_class = self.cls_branches[lvl](hs[lvl])  # (B, 300, 10)
    
    # 回归分支
    tmp = self.reg_branches[lvl](hs[lvl])  # (B, 300, 10)
    # 残差更新：预测 + 参考点
    tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
    tmp[..., 0:2] = tmp[..., 0:2].sigmoid()  # 归一化
    
    # 反归一化到真实坐标（point_cloud_range）
    tmp[..., 0:1] = tmp[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    tmp[..., 1:2] = tmp[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    
    outputs_classes.append(outputs_class)
    outputs_coords.append(tmp)
```

**输出**：
- `outputs_classes`: `(6, B, 300, 10)` - 10类（car/truck/...）
- `outputs_coords`: `(6, B, 300, 10)` - 10维 [x,y,w,l,z,h,sin(yaw),cos(yaw),vx,vy]

**深监督的作用**：
- ✅ 训练时：6层loss累加，帮助梯度流动
- ✅ 推理时：只用最后一层`outputs_classes[-1]`

#### **3.1.5 阶段4：Map解码（第597-615行）**

```python
for lvl in range(6):
    # 分类：对每条线的20个点取平均
    map_outputs_class = self.map_cls_branches[lvl](
        map_hs[lvl].view(bs, map_num_vec, map_num_pts_per_vec, -1).mean(2)
    )  # (B, 100, 4) - 4类（divider/boundary/crossing/bg）
    
    # 回归：预测每个点的坐标
    tmp = self.map_reg_branches[lvl](map_hs[lvl])
    tmp += reference  # 残差更新
    tmp = tmp.sigmoid()
    
    # 转换为box + pts格式
    map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
```

**输出**：
- `map_outputs_classes`: `(6, B, 100, 4)` - 100条线的分类
- `map_outputs_pts_coords`: `(6, B, 100, 20, 2)` - 每条线20个点的xy坐标

#### **3.1.6 阶段5：Motion Head（第617-697行）**

**核心思想**：每个agent预测**多模态轨迹**（6种可能的行为）。

```python
# 1. 扩展为多模态query
motion_query = hs[-1].permute(1, 0, 2)  # (300, B, 256)
mode_query = self.motion_mode_query.weight  # (6, 256)
# 300 agents × 6 modes = 1800 queries
motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)

# 2. agent之间交互（self-attention）
motion_hs = self.motion_decoder(
    query=motion_query, key=motion_query, value=motion_query, ...)
# 学习: "前车减速 → 我也要预测减速"

# 3. agent与地图交互（cross-attention）
map_query = self.lane_encoder(map_hs[-1])  # 提取车道线特征
ca_motion_query = self.motion_map_decoder(
    query=motion_hs, 
    key=map_query,   # ← 用Map Head的输出
    value=map_query, ...)
# 学习: "前方有路口 → 这辆车可能转弯"

# 4. FFN输出轨迹
motion_hs = torch.cat([motion_hs, ca_motion_query], dim=-1)  # 拼接
outputs_traj = self.traj_branches[0](motion_hs)  # (B, 300, 6, 6, 2)
outputs_traj_class = self.traj_cls_branches[0](motion_hs)  # (B, 300, 6)
```

**信息流**：
```
Detection Head (300个agent) ──┐
                              ├─ 扩展为 300×6=1800 queries
Map Head (车道线信息) ─────────┤
                              ↓
                       Motion Decoder
                         ↓        ↓
                    Self-Attn  Cross-Attn(与map)
                         ↓
                  6条多模态轨迹/agent
```

**输出**：
- `outputs_traj`: `(B, 300, 6, 6, 2)` - 300 agents × 6 modes × 6 steps × xy
- `outputs_traj_class`: `(B, 300, 6)` - 每个mode的概率

#### **3.1.7 阶段6：Planning Head（第709-794行）**⭐

**核心思想**：ego query与agent、map交互后，输出3模态自车轨迹。

```python
# 1. 初始化ego query
if self.ego_his_encoder is not None:
    ego_his_feats = self.ego_his_encoder(ego_his_trajs)  # (B, 1, 256)
else:
    ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)

# 2. ego ↔ agent 交互
agent_query = motion_hs.reshape(batch, num_agent, -1)
agent_query = self.agent_fus_mlp(agent_query)  # 降维
agent_query, agent_pos, agent_mask = self.select_and_pad_query(
    agent_query, outputs_coords_bev[-1], agent_conf, ...)  # 选高置信度agent
ego_agent_query = self.ego_agent_decoder(
    query=ego_query, 
    key=agent_query, 
    value=agent_query, ...)
# 学习: "前方有车 → 别撞上"

# 3. ego ↔ map 交互
map_query = self.lane_encoder(map_hs[-1])  # 提取车道线特征
map_query, map_pos, map_mask = self.select_and_pad_query(
    map_query, min_map_pos, map_conf, ...)  # 选高置信度车道线
ego_map_query = self.ego_map_decoder(
    query=ego_agent_query,  # ← 接上一步输出
    key=map_query, 
    value=map_query, ...)
# 学习: "车道在这 → 沿车道开"

# 4. 拼接CAN总线特征
ego_feats = torch.cat([
    ego_his_feats,              # 历史轨迹特征
    ego_map_query,              # 与map交互后的特征
    ego_lcf_feat[..., idx]      # 速度/加速度/转向角等
], dim=-1)  # (B, 1, 2D+2)

# 5. FFN输出3模态轨迹
outputs_ego_trajs = self.ego_fut_decoder(ego_feats)
outputs_ego_trajs = outputs_ego_trajs.reshape(B, 3, 6, 2)  # ⭐
```

**关键设计**：

| 步骤 | 输入 | 操作 | 输出 |
|------|------|------|------|
| 1 | ego_his_trajs | MLP编码 | ego query |
| 2 | ego + agent | Cross-Attention | ego_agent_query |
| 3 | ego_agent + map | Cross-Attention | ego_map_query |
| 4 | ego_map + CAN | Concat | ego_feats |
| 5 | ego_feats | FFN | 3模态轨迹 |

**信息流依赖链**：
```
Detection ─→ Motion ─┐
                     ├─→ Planning (ego query)
       Map ─────────┘    + ego_lcf_feat (CAN)
                          + ego_his_trajs
```

**关键依赖**：
- Planning用的是**预测的Motion和Map**，不是GT
- 端到端可微：Planning loss反向传播，会优化Motion和Map
- 推理时无缝衔接（不需要GT）

#### **3.1.8 最终输出outs（第796-810行）**

```python
outs = {
    # BEV特征（用于时序传递）
    'bev_embed': bev_embed,  # (B, 10000, 256)
    
    # Detection输出
    'all_cls_scores': outputs_classes,     # (6, B, 300, 10)
    'all_bbox_preds': outputs_coords,      # (6, B, 300, 10)
    
    # Motion输出
    'all_traj_preds': outputs_trajs,       # (6, B, 300, 6, 6, 2)
    'all_traj_cls_scores': outputs_trajs_classes,  # (6, B, 300, 6)
    
    # Map输出
    'map_all_cls_scores': map_outputs_classes,     # (6, B, 100, 4)
    'map_all_bbox_preds': map_outputs_coords,      # (6, B, 100, 4)
    'map_all_pts_preds': map_outputs_pts_coords,   # (6, B, 100, 20, 2)
    
    # Planning输出 ⭐
    'ego_fut_preds': outputs_ego_trajs,    # (B, 3, 6, 2)
}
return outs
```

### 3.2 关键设计哲学

#### **3.2.1 任务依赖关系**

```
   ┌──────────┐
   │   BEV    │  共享底层特征
   └────┬─────┘
        │
   ┌────┴────────┬─────────┐
   ↓             ↓         │
检测          地图        │
   │             │         │
   └──→ Motion ←─┘         │
            │              │
            └──→ Planning ←┘
```

**关键观察**：
- BEV是所有任务的共享表示
- Motion依赖Detection（agent信息）和Map（车道线）
- Planning依赖所有上游任务

#### **3.2.2 Query数量的设计**

| 任务 | Query数 | 设计理由 |
|------|---------|---------|
| BEV | 10000 | 稠密表示场景（100×100网格） |
| Detection | 300 | 单帧最多~50物体，留5×冗余 |
| Map | 2000 | 100条线×20点，足够覆盖路口 |
| Motion | 1800 | 300 agents × 6 modes |
| Planning | 1 | 只有一辆自车 |

#### **3.2.3 三种Attention的角色**

| 类型 | 用途 | VAD中的例子 |
|------|------|-------------|
| **Spatial Cross-Attention** | 跨视角（图像→BEV） | BEV Encoder |
| **Self-Attention** | 同质交互 | Motion中agent之间 |
| **Cross-Attention** | 异质交互 | ego↔agent, motion↔map |

#### **3.2.4 深监督（Deep Supervision）**

```python
# 6层decoder各自输出预测
for lvl in range(6):
    outputs_class[lvl] = cls_branches[lvl](hs[lvl])
```

**作用**：
- ✅ 训练时：6层loss累加，帮助梯度流动到浅层
- ✅ 推理时：只用最后一层`[-1]`
- ✅ 避免梯度消失（深层网络的常见问题）

### 3.4 损失计算：感知任务的监督

#### **3.4.1 数据分流：模型输入 vs 监督GT**

`forward_pts_train`接收很多数据，但**只有部分进forward**，其余进loss。

**代码位置**：`VAD.py` 第220-227行

```python
def forward_pts_train(self, pts_feats, gt_bboxes_3d, ..., prev_bev,
                      ego_his_trajs, ego_fut_trajs, ..., gt_attr_labels):
    # ① 只把"模型输入"喂给 forward
    outs = self.pts_bbox_head(
        pts_feats,                      # ✅ 模型输入
        img_metas,                      # ✅ 模型输入
        prev_bev,                       # ✅ 模型输入
        ego_his_trajs=ego_his_trajs,    # ✅ 模型输入
        ego_lcf_feat=ego_lcf_feat,      # ✅ 模型输入
    )
    # ↑ 所有 gt_* 和 ego_fut_* 都没传进去！

    # ② 把"GT"和"预测outs"一起喂给 loss
    loss_inputs = [gt_bboxes_3d, gt_labels_3d,
                   map_gt_bboxes_3d, map_gt_labels_3d,
                   outs,  # ← forward的预测结果
                   ego_fut_trajs, ego_fut_masks, ego_fut_cmd,
                   gt_attr_labels]
    losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
    return losses
```

**数据分流表**：

| 数据 | 进forward？ | 进loss？ | 角色 |
|------|:---:|:---:|------|
| `pts_feats` | ✅ | ❌ | 模型输入（图像特征） |
| `img_metas` | ✅ | ✅ | 输入+坐标变换 |
| `prev_bev` | ✅ | ❌ | 模型输入（历史BEV） |
| `ego_his_trajs` | ✅ | ❌ | 模型输入（自车历史） |
| `ego_lcf_feat` | ✅ | ❌ | 模型输入（CAN总线） |
| `gt_bboxes_3d` | ❌ | ✅ | 检测GT |
| `gt_labels_3d` | ❌ | ✅ | 类别GT |
| `map_gt_bboxes_3d` | ❌ | ✅ | 地图GT |
| `map_gt_labels_3d` | ❌ | ✅ | 地图类别GT |
| `gt_attr_labels` | ❌ | ✅ | 他车轨迹GT |
| `ego_fut_trajs` | ❌ | ✅ | 自车未来轨迹GT |
| `ego_fut_cmd` | ❌ | ✅ | 指令GT（选哪条轨迹监督） |

**判断标准：部署时有没有这个数据？**

```
部署到车上时（推理）：
  ✅ 有的 → 可以做 forward 输入
     - 相机图像、上一帧BEV、自车历史、CAN总线
  ❌ 没有的 → 只能当训练GT
     - 周围车的真实位置 ← 模型要预测的！
     - 真实地图 ← 模型要预测的！
     - 自车未来该怎么走 ← 模型要预测的！
```

**核心逻辑**：模型要**预测**的东西，其真值（GT）不能作为输入，否则就是"抄答案"（数据泄漏）。

#### **3.4.2 Detection 与 Map 是镜像对称的训练**

训练时两条路径完全对称：

```
┌─────────────────── Detection 分支 ───────────────────┐
│ 图片 → BEV → Detection Head → 300个预测框            │
│                                  ↓                    │
│                          匈牙利匹配                   │
│                                  ↓                    │
│            人工标注的3D box（gt_bboxes_3d）           │
│                                  ↓                    │
│       loss_cls + loss_bbox (+ loss_traj 他车轨迹)     │
└───────────────────────────────────────────────────────┘

┌─────────────────── Map 分支（对称）──────────────────┐
│ 图片 → BEV → Map Head → 100条预测polyline            │
│                                  ↓                    │
│                          匈牙利匹配                   │
│                                  ↓                    │
│         HD Map标注的polyline（map_gt_bboxes_3d）      │
│                                  ↓                    │
│   loss_map_cls + loss_map_bbox + loss_map_pts +       │
│   loss_map_dir                                        │
└───────────────────────────────────────────────────────┘
```

**对比方式的差异**（几何形态不同导致）：

| | 3D box（Detection） | map polyline（Map Head） |
|---|---|---|
| **预测什么** | 1个框：[x,y,z,w,l,h,yaw] | 1条线：20个有序点 |
| **分类** | 10类（car/truck/...） | 3类（divider/boundary/crossing） |
| **位置对比** | L1距离（框中心+尺寸） | Chamfer Distance（点集形状） |
| **额外约束** | 朝向角yaw | 线的方向dir |
| **匹配方式** | 匈牙利匹配 | 匈牙利匹配（相同） |

#### **3.4.3 匈牙利匹配（DETR范式核心）**

两者都先做匈牙利匹配，再算loss。

```
问题：模型输出300个预测框（顺序随机）
     人工标注15个真实框
     → 哪个预测对应哪个GT？

解法：�and牙利算法（linear_sum_assignment）
     → 找到预测↔GT的最优配对
     → 配对的算回归loss，没配上的算"背景"分类loss
```

map同理（`_map_get_target_single`，第1100行）：
```
模型输出100条线（顺序随机）
HD Map标注8条真实线
→ 匈牙利匹配找配对
```

#### **3.4.4 Map的4个损失（loss代码第1421-1464行）**

Map Head预测与HD Map标注对比，不是单个loss，而是4个一起：

```python
# ① 分类损失：预测的线"是哪类" vs 标注的类别
loss_cls = self.loss_map_cls(cls_scores, labels, ...)

# ② 框回归损失：预测的外接框 vs 标注的外接框
loss_bbox = self.loss_map_bbox(bbox_preds, normalized_bbox_targets, ...)

# ③ 点序列损失（核心）：预测的20个点 vs 标注的20个点
loss_pts = self.loss_map_pts(pts_preds, normalized_pts_targets, ...)
#   用 Chamfer Distance 比较两条 polyline 的形状

# ④ 方向损失：预测线的走向 vs 标注线的走向
pts_targets_dir = pts_targets[1:] - pts_targets[:-1]  # 相邻点方向向量
```

**Chamfer Distance（map特有）**：

```
预测线:  P = [p1, p2, ..., p20]
标注线:  G = [g1, g2, ..., g20]

Chamfer = Σ(每个预测点到最近标注点的距离)
        + Σ(每个标注点到最近预测点的距离)
```

**为什么不用逐点L1？**
- 点的"起点"可能不同（预测从左端，标注从右端）
- Chamfer只看"形状是否重合"，不要求点的顺序一一对应

#### **3.4.5 GT的两类作用：感知监督 + 间接约束Planning**

**关键发现**：3D box GT 和 map GT 不仅监督感知，还间接参与Planning约束——且两者完全对称。

```
3D box GT ──监督──→ Detection/Motion ──预测──→ agent_preds, agent_fut_preds
                                                    │
                                                    ↓
                                            loss_plan_col（碰撞约束）

地图 GT ────监督──→ Map Head ──────────预测──→ lane_preds
                                                    │
                                                    ↓
                                      loss_plan_bound（边界约束）
                                      loss_plan_dir（方向约束）
```

**Planning约束用的是Head的预测，不是GT**（loss代码第1148-1169行）：

```python
loss_plan_bound = self.loss_plan_bound(
    ego_fut_preds[ego_fut_cmd==1],
    lane_preds,          # ← Map Head 预测，不是GT！
    lane_score_preds, ...)

loss_plan_col = self.loss_plan_col(
    ego_fut_preds[ego_fut_cmd==1],
    agent_preds,         # ← Detection 预测
    agent_fut_preds,     # ← Motion 预测
    ...)
```

**为什么训练时也用预测而非GT？**
- 推理时没有GT，只有带误差的预测
- 训练时若用"完美GT"约束，推理时遇到"带误差预测"会失效（训练推理不一致）
- 故意用预测算约束 → Planning学会"在不完美感知下也能规划"

**GT作用对照表**：

| 维度 | 3D box GT | 地图 GT |
|------|-----------|---------|
| **监督哪个Head** | Detection + Motion | Map |
| **感知任务损失** | loss_cls, loss_bbox, loss_traj | loss_map_cls, loss_map_pts, loss_map_dir |
| **该Head预测什么** | 他车位置/轨迹 | 车道线 |
| **预测如何用于Planning** | 算碰撞约束 | 算边界+方向约束 |
| **GT直接进Planning loss？** | ❌ 否 | ❌ 否 |

**结论**：地图GT和3D box GT是**镜像对称**的中间监督信号——一个管"路在哪"，一个管"车在哪"，最后都为规划服务，但都隔着一层Head预测。

### 3.5 BEV Encoder详解

**BEV Query 的物理含义**：
```
自车中心为原点，俯视图划分为 200×200 网格
每个网格 = 0.15m × 0.15m（根据point_cloud_range计算）

例如：
BEV[0, 0]     → 自车左后方 (-15m, -30m)
BEV[100, 100] → 自车正前方 (0m, 0m)
BEV[199, 199] → 自车右前方 (15m, 30m)
```

#### **3.5.1 Spatial Cross-Attention**

**代码位置**：`modules/spatial_cross_attention.py`

**核心机制**：（待补充）

---

## 附录 A：完整数据流维度变化

> 以 **VAD_tiny_e2e** 配置为基准（`bev_h=bev_w=100`，`_dim_=256`，`num_query=300`，`map_num_vec=100`，`map_num_pts_per_vec=20`，`fut_ts=6`，`fut_mode=6`，`ego_fut_mode=3`）

### A.1 输入数据维度（一个batch）

```python
# ============================================================
# 【类别1：图像】
# ============================================================
img:                  (B, queue=4, 6, 3, 384, 640)  # 历史4帧+当前帧
                      # B=batch, 6=相机数, 3=RGB
                      # 注：训练时queue=4（含当前帧），推理时单帧

# ============================================================
# 【类别2：3D检测GT】
# ============================================================
gt_bboxes_3d:         (B, N_obj, 9)    # N_obj 各样本不同（动态）
                      # 9维: [x,y,z, w,l,h, yaw, vx,vy]
gt_labels_3d:         (B, N_obj)       # 类别ID 0-9
gt_attr_labels:       (B, N_obj, 34)   # 属性（含未来轨迹GT）

# ============================================================
# 【类别3：地图GT】
# ============================================================
map_gt_bboxes_3d:     (B, N_map, 20, 2)  # N_map各样本不同
                      # 20=每条polyline的点数, 2=xy
map_gt_labels_3d:     (B, N_map)         # 0:divider, 1:boundary, 2:crossing

# ============================================================
# 【类别4：自车数据 - 免费午餐】
# ============================================================
ego_his_trajs:        (B, 1, 2, 3)    # 历史2步offset (过去1秒，0.5s/步), 3=xyz
ego_fut_trajs:        (B, 1, 6, 2)    # 未来6步offset (未来3秒，0.5s/步), 2=xy
ego_fut_masks:        (B, 1, 6)       # 6步的有效性
ego_fut_cmd:          (B, 1, 1, 3)    # one-hot: [Right, Left, Straight]
ego_lcf_feat:         (B, 1, 9)       # 9维低频特征

# ============================================================
# 【类别5：图像元信息】
# ============================================================
img_metas:            list of dict    # 含can_bus[18]、lidar2img矩阵等
```

### A.2 模型前向传播维度变化

```
┌──────────────────────────────────────────────────────────────────┐
│ 阶段0: 输入                                                       │
│   img: (B, 4, 6, 3, 384, 640)   # 4帧历史                        │
└──────────────────────────────────────────────────────────────────┘
                            ↓ 拆出当前帧+历史帧
┌──────────────────────────────────────────────────────────────────┐
│ 阶段1: 历史BEV提取（obtain_history_bev）                          │
│   for t in range(4):                                             │
│     当前帧图像 (B, 6, 3, 384, 640)                               │
│         ↓ reshape: (B*6, 3, 384, 640)                            │
│         ↓ ResNet50 stage4                                         │
│     ResNet特征 (B*6, 2048, 12, 20)                               │
│         ↓ FPN压缩通道                                              │
│     FPN特征 (B*6, 256, 12, 20)                                   │
│         ↓ reshape: (B, 6, 256, 12, 20)                           │
│     img_feats (B, 6, 256, 12, 20)                                │
│         ↓ BEV Encoder (only_bev=True)                            │
│     prev_bev (B, 100*100=10000, 256)                             │
│   最终输出 prev_bev: (B, 10000, 256)                             │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│ 阶段2: 当前帧图像特征提取（extract_feat）                          │
│   img_curr (B, 6, 3, 384, 640)                                   │
│       ↓ ResNet50 + FPN                                           │
│   img_feats: list[Tensor]                                        │
│     img_feats[0]: (B, 6, 256, 12, 20)  # 只有1个尺度（tiny配置）  │
└──────────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│ 阶段3: BEV Encoder（VAD_head.py内部）                             │
│   输入：                                                           │
│     img_feats[0]:  (B, 6, 256, 12, 20)  # 当前帧2D特征           │
│     prev_bev:      (B, 10000, 256)       # 上一帧BEV             │
│     bev_queries:   (10000, 256)          # 可学习参数             │
│                                                                   │
│   3.1 Temporal Self-Attention:                                    │
│       bev_queries + prev_bev → 时序融合后的query                  │
│       输出: (B, 10000, 256)                                       │
│                                                                   │
│   3.2 Spatial Cross-Attention:                                    │
│       每个BEV query点 → 投影到6个相机 → 采样2D特征                  │
│       输入: (B, 10000, 256) + (B, 6, 256, 12, 20)                │
│       输出: bev_embed (B, 10000, 256)                            │
└──────────────────────────────────────────────────────────────────┘
                            ↓ bev_embed
        ┌───────────────────┼───────────────────┬─────────────────┐
        ↓                   ↓                   ↓                 ↓
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Det Head     │   │ Motion Head  │   │ Map Head     │   │ Plan Head    │
├──────────────┤   ├──────────────┤   ├──────────────┤   ├──────────────┤
│ 300个query   │   │ 复用Det query│   │ 100*20个     │   │ 1个ego query │
│ ↓ Decoder    │   │ ↓ Motion     │   │ map query    │   │ ↓ Decoder    │
│ ↓ FFN        │   │   Decoder    │   │ ↓ Decoder    │   │              │
│              │   │              │   │              │   │              │
│ 输出:         │   │ 输出:         │   │ 输出:         │   │ 输出:         │
│ cls (B,300,10)│  │ traj         │   │ map_cls      │   │ ego_fut      │
│ box (B,300,9)│   │ (B,300,6,6,2)│   │ (B,2000,3)   │   │ (B,3,6,2)    │
│              │   │ traj_cls     │   │ map_pts      │   │              │
│              │   │ (B,300,6)    │   │ (B,100,20,2) │   │              │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

### A.3 各Head输出维度详解

#### **Detection Head 输出**

```python
all_cls_scores:    (n_dec_layers=6, B, 300, 10)
                   # 6层decoder各自输出（深监督）
                   # 300个object query
                   # 10类（car/truck/.../traffic_cone）

all_bbox_preds:    (n_dec_layers=6, B, 300, 10)
                   # 10维 = [x,y,w,l,z,h, sin(yaw), cos(yaw), vx, vy]
                   # 注：实际是10维编码格式，解码后才是7维[x,y,z,w,l,h,yaw]

# 推理时取最后一层 + 过滤低置信度
boxes_3d:          (N_keep, 7)    # N_keep = 阈值后保留数（通常<300）
                                  # 7维: [x,y,z, w,l,h, yaw]
scores_3d:         (N_keep,)      # 置信度
labels_3d:         (N_keep,)      # 类别ID
```

#### **Motion Head 输出**

```python
# 多模态他车未来轨迹
all_traj_preds:    (n_dec_layers=6, B, 300, 6, 6, 2)
                   # 300个object（与det共享）
                   # 6个模态（fut_mode=6，多模态预测）
                   # 6步未来（fut_ts=6，3秒）
                   # 2维xy offset

all_traj_cls:      (n_dec_layers=6, B, 300, 6)
                   # 每个模态的概率

# 推理时
trajs_3d:          (N_keep, 6, 6, 2)   # 配对到检测框上的轨迹
```

**关键说明**：每个agent输出**6条候选轨迹**，每条对应不同行为模式（直行/左转/右转/加速/减速/...）。

#### **Map Head 输出**

```python
# 矢量化地图元素
all_map_cls:       (n_dec_layers=6, B, 100, 4)
                   # 100个polyline query
                   # 4 = 3类(divider/boundary/crossing) + 1背景

all_map_pts:       (n_dec_layers=6, B, 100, 20, 2)
                   # 每条polyline 20个点（固定采样）
                   # 2维xy

all_map_bbox:      (n_dec_layers=6, B, 100, 4)
                   # 每条polyline的外接框 [x_min, y_min, x_max, y_max]

# 推理时
map_pts:           (N_map_keep, 20, 2)
map_scores:        (N_map_keep,)
map_labels:        (N_map_keep,)
```

#### **Planning Head 输出** ⭐核心

```python
ego_fut_preds:     (B, 3, 6, 2)
                   # 3 = ego_fut_mode（左转/右转/直行 各1条）
                   # 6 = fut_ts（未来6步=3秒）
                   # 2 = xy offset（每步相对位移）

# 推理时根据导航指令选取对应轨迹
ego_fut_cmd_idx = argmax(ego_fut_cmd)  # 0/1/2
ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]  # (6, 2)

# 转累积坐标（送入控制器）
ego_fut_pred = ego_fut_pred.cumsum(dim=0)  # (6, 2) 绝对位置
```

### A.4 各阶段维度变化速查表

| 阶段 | 张量 | 维度 | 物理含义 |
|------|------|------|---------|
| **输入** | img | `(B, 4, 6, 3, 384, 640)` | 4帧×6相机×RGB |
| **ResNet后** | resnet_feat | `(B*6, 2048, 12, 20)` | stage4高层语义 |
| **FPN后** | fpn_feat | `(B*6, 256, 12, 20)` | 通道压缩到256 |
| **重塑后** | img_feats | `(B, 6, 256, 12, 20)` | 6相机分开 |
| **BEV Query** | bev_queries | `(10000, 256)` | 100×100网格 |
| **BEV特征** | bev_embed | `(B, 10000, 256)` | 俯视图统一表示 |
| **Det输出** | boxes_3d | `(N_keep, 7)` | 检测框 |
| **Motion输出** | trajs_3d | `(N_keep, 6, 6, 2)` | 多模态他车轨迹 |
| **Map输出** | map_pts | `(N_map_keep, 20, 2)` | 矢量化地图 |
| **Plan输出** | ego_fut_pred | `(6, 2)` | **最终规划轨迹** ⭐ |

### A.5 维度变化的关键转折点

```
┌───────────────────────────────────────────────────────┐
│ 转折1: 像素 → 语义                                     │
│   (B, 6, 3, 384, 640) → (B, 6, 256, 12, 20)          │
│   - 空间分辨率↓ 32倍（384/12=32）                      │
│   - 通道数↑：3 → 256（语义维度）                       │
│   - 由 ResNet50 + FPN 完成                            │
└───────────────────────────────────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────┐
│ 转折2: 透视 → BEV（最关键的视角转换）                   │
│   (B, 6, 256, 12, 20) → (B, 10000, 256)              │
│   - 6个相机视角 → 1个统一俯视图                        │
│   - 12×20=240格 × 6相机 = 1440格 → 100×100=10000格    │
│   - 由 BEV Encoder 的 Spatial Cross-Attention 完成    │
│   - 关键：从图像坐标系跳到自车中心坐标系                │
└───────────────────────────────────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────┐
│ 转折3: 稠密BEV → 稀疏query                              │
│   bev_embed (B, 10000, 256) → 各种query              │
│   - Det query: 300个                                  │
│   - Map query: 100×20=2000个                         │
│   - Ego query: 1个（v1）/ 4096个（v2词表）             │
│   - 由 Cross-Attention（query与bev_embed交互）完成     │
└───────────────────────────────────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────┐
│ 转折4: query → 任务输出                                 │
│   各query → FFN → 具体预测值                           │
│   - 检测框、轨迹、地图点、规划轨迹                      │
│   - 由各自的 prediction head（FFN）完成                │
└───────────────────────────────────────────────────────┘
```

### A.6 训练 vs 推理的维度差异

| 阶段 | 训练 | 推理 |
|------|------|------|
| **图像输入** | `(B, queue=4, 6, 3, H, W)` 含历史 | `(B, 6, 3, H, W)` 单帧 |
| **历史BEV** | 每个batch临时计算 | 跨帧缓存（`prev_frame_info`） |
| **Decoder层数** | 6层全部输出（深监督） | 取最后一层 |
| **Det query** | 300全部参与loss计算 | 经阈值过滤（保留~10-30个） |
| **Plan输出** | `(B, 3, 6, 2)` 3个模态都算loss | `(6, 2)` 按cmd选1个 |

### A.7 数值约束（point_cloud_range）

```python
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
                    [x_min, y_min, z_min, x_max, y_max, z_max]
```

**物理意义**：
- BEV覆盖：x方向±15m（左右），y方向±30m（前后），z方向±2m（高度）
- BEV分辨率：30m / 100格 = **0.3m/格**（VAD_tiny）
  - VAD_base：60m / 200格 = **0.3m/格**（覆盖更大但分辨率不变）
- 超出范围的物体：训练时被`CustomObjectRangeFilter`过滤

**所有输出坐标都在这个范围内**，且坐标系是**当前帧lidar坐标系**（自车中心）。

