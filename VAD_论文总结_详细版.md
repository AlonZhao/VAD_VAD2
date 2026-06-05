# VAD: Vectorized Scene Representation for Efficient Autonomous Driving 论文详细总结

## 论文基本信息
- **标题**: VAD: Vectorized Scene Representation for Efficient Autonomous Driving
- **机构**: Huazhong University of Science & Technology, Horizon Robotics
- **作者**: Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, Xinggang Wang
- **发表**: arXiv:2303.12077v3 [cs.RO] 24 Aug 2023
- **代码**: https://github.com/hustvl/VAD

---

## 各章节详细内容总结

### 1. Abstract & Introduction

#### 1.1 研究背景与动机

**传统方法的局限性**：

自动驾驶系统需要对周围环境进行全面理解以实现可靠的轨迹规划。现有的端到端方法主要面临两类问题：

1. **直接端到端方法**（如 ALVINN, Behavioral Cloning）
   - 直接从传感器数据输出控制信号
   - **缺陷**: 缺乏可解释性（lack interpretability），训练困难（difficult to optimize）
   - 无法提供中间表示，调试和改进困难

2. **基于栅格化表示的方法**（如 ST-P3, UniAD）
   - 使用密集的栅格化场景表示：
     - **Semantic Map**: 语义地图
     - **Occupancy Map**: 占用栅格图
     - **Flow Map**: 光流/运动流场图
     - **Cost Map**: 代价地图
   - **缺陷**: 
     - 计算密集（computationally intensive）
     - 缺失实例级结构信息（misses instance-level structure information）
     - 需要复杂的后处理步骤（hand-designed post-processing）

#### 1.2 VAD 核心思想

**Vectorized Scene Representation（矢量化场景表示）**：

VAD 将驾驶场景建模为完全矢量化的表示，包括：

1. **Vectorized Map（矢量化地图）**
   - **Boundary Vectors**: 道路边界向量，定义可行驶区域（drivable area）
   - **Lane Vectors**: 车道线向量，提供车道方向（lane direction）和交通流信息（traffic flow）
   - 表示形式: $\hat{V}_m \in \mathbb{R}^{N_m \times N_p \times 2}$
     - $N_m$: 地图向量数量
     - $N_p$: 每个向量的点数
     - 2: (x, y) 坐标

2. **Agent Motion Vectors（智能体运动向量）**
   - 表示其他交通参与者的未来轨迹
   - 多模态预测（multi-modality prediction）
   - 表示形式: $\hat{V}_a \in \mathbb{R}^{N_a \times N_k \times T_f \times 2}$
     - $N_a$: 检测到的 agent 数量
     - $N_k$: 模态数量（每个 agent 有多种可能的未来行为）
     - $T_f$: 未来时间步数

3. **Ego Vector（自车轨迹向量）**
   - 规划的自车未来轨迹
   - 表示形式: $\hat{V}_{ego} \in \mathbb{R}^{T_f \times 2}$

#### 1.3 矢量化表示的优势

1. **结构化信息（Structured Information）**
   - 矢量化地图保留道路拓扑结构
   - 车道方向可显式约束规划
   - 缩小轨迹搜索空间（narrow down trajectory search space）

2. **实例级约束（Instance-Level Constraints）**
   - 每个 agent 的运动向量提供独立的碰撞约束
   - 每个地图元素提供独立的边界约束
   - 比密集栅格图更精确

3. **计算效率（Computational Efficiency）**
   - 避免密集预测（如 200×200 的 BEV 栅格图）
   - 稀疏表示（sparse representation）
   - 关键对于实际部署（critical for real-world deployment）

#### 1.4 VAD 的双重信息利用机制

**隐式利用（Implicit Utilization）**：
- 通过 **Query Interaction** 机制
- Agent queries $Q_a$ 和 map queries $Q_m$ 与 ego query $Q_{ego}$ 交互
- 通过 cross-attention 提取特征级信息

**显式利用（Explicit Utilization）**：
- 三个几何约束（geometric constraints）:
  1. **Ego-Agent Collision Constraint**: 维持自车与其他车辆的安全距离
  2. **Ego-Boundary Overstepping Constraint**: 防止越过道路边界
  3. **Ego-Lane Direction Constraint**: 约束运动方向与车道方向一致

#### 1.5 主要贡献（Key Contributions）

1. **范式创新**
   - 提出 VAD，首个完全基于矢量化表示的端到端自动驾驶框架
   - 消除计算密集的栅格化表示和手工设计的后处理步骤

2. **方法创新**
   - 隐式 + 显式双重利用矢量化场景信息
   - Query interaction 机制实现特征级信息流动
   - 三个实例级矢量化规划约束提升安全性

3. **性能突破**
   - **精度提升**: 相比 UniAD，平均规划误差从 1.03m 降至 0.72m（↓30.1%）
   - **安全提升**: 平均碰撞率从 0.31% 降至 0.22%（↓29.0%）
   - **速度提升**: 推理速度从 1.8 FPS 提升至 4.5 FPS（↑2.5×）
   - **轻量版本**: VAD-Tiny 达到 16.8 FPS（↑9.3×），性能仍具竞争力

---

### 2. Related Work（相关工作）

#### 2.1 Perception for Autonomous Driving

**Camera-based 3D Object Detection（基于相机的 3D 目标检测）**：

1. **Query-based Methods**
   - **DETR3D**: 使用 3D queries 采样图像特征，无需 NMS（Non-Maximum Suppression）
   - **PETR**: 引入 3D positional encoding 到图像特征，通过 attention 机制学习物体特征

2. **BEV-based Methods（Bird's Eye View）**
   - **LSS (Lift-Splat-Shoot)**: 先驱工作，通过深度预测将透视图特征投影到 BEV
   - **BEVFormer**: 
     - 提出 spatial cross-attention 和 temporal self-attention
     - 仅用相机输入达到优异的检测性能
     - VAD 采用类似的 BEV 特征编码方式

**Online HD Map Construction（在线高精地图构建）**：

1. **Rasterized Methods**
   - **FIERY, BEVerse**: 使用 BEV 特征图预测密集的地图分割
   - **HDMapNet**: 先分割再后处理转为矢量化地图

2. **Vectorized Methods（矢量化方法）**
   - **VectorMapNet**: 自回归方式预测地图元素点
   - **MapTR**: 
     - 识别地图实例点的排列不变性（permutation invariance）
     - 可同时预测所有地图元素
     - VAD 借鉴其设计
   - **LaneGAP**: 以路径方式建模车道图（path-wise manner），保留车道连续性

#### 2.2 Motion Prediction

**传统方法（使用 GT 感知结果）**：

1. **Rasterized Scene Rendering**
   - 将驾驶场景渲染为 BEV 图像
   - 使用 CNN 预测未来运动
   - 例如: CoverNet, MultiPath

2. **Vectorized Representation**
   - **VectorNet**: 使用矢量化表示 + GNN
   - **LaneGCN**: 车道图 + Graph Neural Network
   - **mmTransformer, Scene Transformer**: Transformer-based 方法

**端到端方法（Joint Perception and Prediction）**：

1. **Occupancy-based**
   - **FIERY, BEVerse**: 将未来运动视为密集占用和光流，而非 agent-level waypoints
   - **HOPE**: 层次化时空网络预测占用流

2. **Instance-based**
   - **ViP3D**: 基于跟踪结果和 HD map 预测未来运动
   - **PIP**: 
     - 提出动态 agent 和静态矢量化地图的交互方案
     - 不依赖 HD map 达到 SOTA
     - **VAD 受其启发**，通过 agent-map interaction 学习运动

#### 2.3 End-to-End Planning

**Reinforcement Learning 方法**：
- Learning by Cheating, GRI
- 有前景但训练复杂

**Cost Map-based 方法**：
- **MP3, ST-P3, UniAD**: 
  - 从感知/预测结果或学习模块构建密集代价图
  - 使用手工规则选择最小代价轨迹
  - **问题**: 构建代价图计算密集，手工规则带来鲁棒性和泛化问题

**UniAD（当前 SOTA）**：
- Goal-oriented 精神，有效整合各前序任务信息辅助规划
- 在感知、预测、规划上达到显著性能
- **局限**: 依赖栅格化表示，推理速度慢

**PlanT**：
- 使用 GT 感知结果
- 以 object-level representation 编码场景进行规划

**VAD 的差异化**：
- 探索矢量化场景表示用于规划
- 无需密集地图或手工设计后处理步骤
- 端到端可微分训练

---

### 3. Method（方法详解）

#### 3.1 Overall Architecture（总体架构）

VAD 采用四阶段流水线：

**输入**: 多帧多视角图像 $\{I_t^{cam_i}\}$
- $t$: 时间戳（历史 + 当前）
- $cam_i$: 6 个环视相机

**Stage 1: Image Feature Extraction**
- **Backbone**: ResNet50 或 ResNet101
- 输出: 多尺度图像特征 $F_{img}$

**Stage 2: BEV Feature Encoding**
- **BEV Queries**: $Q_{bev} \in \mathbb{R}^{H_{bev} \times W_{bev} \times C}$
- 通过 **Spatial Cross-Attention** 从图像特征采样
- 通过 **Temporal Self-Attention** 融合历史 BEV 特征
- 输出: BEV 特征图 $F_{bev}$

**Stage 3: Vectorized Scene Learning**
- **Map Module**: $Q_m$ queries → 矢量化地图 $\hat{V}_m$
- **Motion Module**: $Q_a$ queries → agent 检测 + 多模态轨迹 $\hat{V}_a$

#### 3.2 Vectorized Scene Learning（矢量化场景学习）

##### 3.2.1 Vectorized Map Learning

**目标**: 从 BEV 特征中检测并矢量化地图元素

**地图元素类别**:
1. **Lane Divider（车道分隔线）**: 提供方向信息
2. **Road Boundary（道路边界）**: 指示可行驶区域
3. **Pedestrian Crossing（人行横道）**: 行人过街区域

**网络结构**:
- 初始化 $N_m$ 个可学习的 map queries: $Q_m \in \mathbb{R}^{N_m \times C}$
- 通过 **Deformable Attention** 与 BEV 特征交互
- 每个 query 预测:
  - 地图向量: $N_p$ 个有序点 $(x, y)$
  - 类别得分: 3 类 + background

**输出表示**:
$$\hat{V}_m \in \mathbb{R}^{N_m \times N_p \times 2}$$

其中:
- $N_m$: map query 数量（默认 100）
- $N_p$: 每个向量的点数（默认 20）
- 2: BEV 坐标 $(x, y)$

**损失函数**:

1. **回归损失（Regression Loss）**:
$$\mathcal{L}_{map}^{reg} = \frac{1}{N_m} \sum_{i=1}^{N_m} \sum_{j=1}^{N_p} \|p_{ij} - \hat{p}_{ij}\|_1$$
   - 使用 Manhattan 距离（L1 loss）
   - 点级别监督

2. **分类损失（Classification Loss）**:
$$\mathcal{L}_{map}^{cls} = \text{FocalLoss}(s_m, \hat{s}_m)$$
   - Focal Loss 处理类别不平衡

3. **总地图损失**:
$$\mathcal{L}_{map} = \lambda_1 \mathcal{L}_{map}^{reg} + \lambda_2 \mathcal{L}_{map}^{cls}$$

**关键设计**:
- 采用类似 MapTR 的匹配策略（Hungarian matching）
- 地图向量的排列不变性（permutation invariance）

##### 3.2.2 Vectorized Agent Motion Learning

**两阶段流程**: 检测（Detection） + 运动预测（Motion Prediction）

**阶段 1: Agent Detection**

- 初始化 $N_a$ 个 agent queries: $Q_a \in \mathbb{R}^{N_a \times C}$
- 通过 Deformable Attention 从 $F_{bev}$ 提取特征
- 预测 agent 属性:
  - 中心位置: $(x, y, z)$
  - 朝向角: $\theta$
  - 尺寸: $(l, w, h)$
  - 速度: $(v_x, v_y)$
  - 类别得分: 10 类（car, truck, bus, ...）

**阶段 2: Agent-Agent & Agent-Map Interaction**

目的: 丰富 agent features 以进行更准确的运动预测

1. **Agent-Agent Interaction**:
$$Q_a^{aa} = \text{SelfAttention}(Q_a, Q_a, Q_a)$$
   - 捕获 agent 之间的社交关系（social interaction）
   - 例如: 前车减速影响后车

2. **Agent-Map Interaction**:
$$Q_a^{am} = \text{CrossAttention}(Q_a^{aa}, Q_m, Q_m)$$
   - agent queries 作为 query
   - map queries 作为 key 和 value
   - 捕获 agent 与道路结构的关系（如车辆倾向于沿车道行驶）

**阶段 3: Multi-Modal Trajectory Prediction**

从交互后的 $Q_a^{am}$ 预测多模态未来轨迹:

$$\hat{V}_a \in \mathbb{R}^{N_a \times N_k \times T_f \times 2}$$

其中:
- $N_k$: 模态数量（默认 6），每个模态代表一种驾驶意图（如左转、右转、直行）
- $T_f$: 未来时间步（默认 6，对应 3 秒 @ 2Hz）
- 每个模态有置信度得分: $p_k \in [0, 1]$, $\sum_k p_k = 1$

**损失函数**:

1. **检测损失**:
$$\mathcal{L}_{agent} = \mathcal{L}_{reg}^{bbox} + \mathcal{L}_{cls}^{agent}$$

2. **运动预测损失（Min-of-N）**:
$$\mathcal{L}_{motion} = \frac{1}{N_a} \sum_{i=1}^{N_a} \min_{k \in [1, N_k]} \|V_a^i - \hat{V}_{a,k}^i\|_1$$
   - 仅对最佳模态（最接近 GT 的那个）计算损失
   - 鼓励多样性（diversity）

3. **模态分类损失**:
$$\mathcal{L}_{cls}^{mode} = \text{FocalLoss}(p_k, \hat{p}_k)$$

4. **总运动损失**:
$$\mathcal{L}_{motion\_total} = \mathcal{L}_{agent} + \lambda_m \mathcal{L}_{motion} + \lambda_c \mathcal{L}_{cls}^{mode}$$

#### 3.3 Planning via Interaction（交互式规划）

##### 3.3.1 Ego Query Initialization

**Ego Query**: $Q_{ego} \in \mathbb{R}^{1 \times C}$
- 可学习的嵌入（learnable embedding）
- 通过与场景 queries 交互学习隐式场景特征

**可选输入**: Ego Status $s_{ego}$
- 当前速度: $v$
- 加速度: $a$
- 角速度: $\omega$
- 可通过 MLP 编码后与 $Q_{ego}$ 拼接

**注意**: 论文主实验中**不使用** ego status，以避免开环评估中的 shortcut learning（捷径学习）

##### 3.3.2 Ego-Agent Interaction

**目的**: 让 ego query 感知其他交通参与者的位置和运动

**Transformer Decoder 结构**:

$$Q_{ego}^{'} = \text{TransformerDecoder}(q, k, v, q_{pos}, k_{pos})$$

其中:
- $q = Q_{ego}$: ego query 作为 query
- $k = v = Q_a$: agent queries 作为 key 和 value
- $q_{pos} = \text{PE}_1(p_{ego})$: ego 位置编码
- $k_{pos} = \text{PE}_1(p_a)$: agent 位置编码

**位置编码的作用**:
- 提供 ego 与各 agent 的相对位置关系
- 例如: 前方 10m 的车 vs. 左侧 5m 的车，对规划的影响不同

**输出**: 更新后的 ego query $Q_{ego}^{'}$，包含动态场景信息

##### 3.3.3 Ego-Map Interaction

**目的**: 让 ego query 感知道路结构（车道、边界）

$$Q_{ego}^{''} = \text{TransformerDecoder}(q, k, v, q_{pos}, k_{pos})$$

其中:
- $q = Q_{ego}^{'}$: 与 agent 交互后的 ego query
- $k = v = Q_m$: map queries 作为 key 和 value
- $q_{pos} = \text{PE}_2(p_{ego})$: ego 位置编码（不同的 MLP）
- $k_{pos} = \text{PE}_2(p_m)$: map 元素位置编码

**为什么用不同的位置编码器 $\text{PE}_1$ 和 $\text{PE}_2$**:
- Agent 和 map 的空间分布特性不同
- Agent 是点状（point-like），map 是线状（line-like）
- 独立的 PE 可以更好地捕获各自特性

**输出**: 最终的 ego query $Q_{ego}^{''}$，包含动态 + 静态场景信息

##### 3.3.4 Planning Head

**输入**:
1. 交互后的 ego features: $f_{ego} = [Q_{ego}^{'}, Q_{ego}^{''}, s_{ego}]$（可选 ego status）
2. 高层驾驶指令: $cmd \in \{\text{左转, 右转, 直行}\}$

**为什么需要驾驶指令**:
- VAD 是 HD-map-free 规划，没有全局路径信息
- 驾驶指令提供导航意图（navigation intent）
- 在交叉口等场景决定转向

**网络结构**:
- 简单的 MLP-based decoder
- 指令通过 embedding layer 编码后与 $f_{ego}$ 拼接

**输出**: 自车规划轨迹
$$\hat{V}_{ego} \in \mathbb{R}^{T_f \times 2}$$

#### 3.4 Vectorized Planning Constraint（矢量化规划约束）

**核心思想**: 利用矢量化场景表示，在训练阶段对规划轨迹施加实例级几何约束

**三个约束的互补性**:
- **Collision Constraint**: 动态安全（与其他车辆）
- **Boundary Constraint**: 静态安全（保持在道路内）
- **Direction Constraint**: 合理性约束（符合交通规则）

##### 3.4.1 Ego-Agent Collision Constraint（碰撞约束）

**动机**: 
- 传统方法使用密集占用图判断碰撞，计算昂贵
- VAD 直接基于矢量化轨迹计算碰撞风险

**设计原理**:
- **横向（Lateral）**: 多车可以并排行驶，安全距离较小
- **纵向（Longitudinal）**: 需要更大的刹车距离

**算法流程**:

1. **过滤低置信度 agent**: 
   - 置信度阈值 $\tau_a = 0.5$
   - 对多模态预测，选择最高置信度的模态

2. **对每个未来时刻 $t \in [1, T_f]$**:
   - 计算 ego 位置 $p_{ego}^t$ 到所有 agent 位置 $p_a^t$ 的距离
   - 找到最近的 agent（在一定范围 $\rho_a = 3.0m$ 内）
   - 分别计算横向和纵向距离: $d_X^t$, $d_Y^t$

3. **计算损失**:
   - 对每个方向 $i \in \{X, Y\}$:
   $$\mathcal{L}_{col}^{i,t} = \begin{cases}
   \delta_i - d_i^t & \text{if } d_i^t < \delta_i \\
   0 & \text{otherwise}
   \end{cases}$$
   
   其中:
   - $\delta_X = 1.5m$: 横向安全阈值
   - $\delta_Y = 3.0m$: 纵向安全阈值（2倍横向）

4. **总碰撞损失**:
$$\mathcal{L}_{col} = \frac{1}{T_f} \sum_{t=1}^{T_f} \left(\mathcal{L}_{col}^{X,t} + \mathcal{L}_{col}^{Y,t}\right)$$

**效果**: 
- 推动规划轨迹远离其他车辆
- 纵向保持更大安全距离（符合驾驶习惯）

##### 3.4.2 Ego-Boundary Overstepping Constraint（边界越界约束）

**目的**: 确保规划轨迹在可行驶区域内

**算法流程**:

1. **过滤地图元素**:
   - 仅考虑 **road boundary** 类别
   - 置信度阈值 $\tau_m = 0.5$

2. **对每个未来时刻 $t$**:
   - 计算 ego waypoint $p_{ego}^t$ 到所有 boundary vectors 的最短距离
   - $d_{bd}^t = \min_{b \in \text{Boundaries}} \text{dist}(p_{ego}^t, b)$
   - 这里 $\text{dist}$ 是点到折线段的距离

3. **损失函数**:
$$\mathcal{L}_{bd}^t = \begin{cases}
\delta_{bd} - d_{bd}^t & \text{if } d_{bd}^t < \delta_{bd} \\
0 & \text{otherwise}
\end{cases}$$

其中 $\delta_{bd} = 1.0m$

4. **总边界损失**:
$$\mathcal{L}_{bd} = \frac{1}{T_f} \sum_{t=1}^{T_f} \mathcal{L}_{bd}^t$$

**效果**:
- 推动轨迹远离边界
- 类似软约束的 "力场"（repulsive force field）

##### 3.4.3 Ego-Lane Directional Constraint（车道方向约束）

**先验假设**: 车辆运动方向应与所在车道方向一致

**算法流程**:

1. **过滤车道线**:
   - 仅考虑 **lane divider** 类别
   - 置信度阈值 $\tau_m = 0.5$

2. **对每个未来时刻 $t$**:
   - 找到距离 ego waypoint $p_{ego}^t$ 最近的车道向量 $\hat{v}_m^t$（在 $\rho_{dir} = 2.0m$ 范围内）
   - 车道向量: 由两个连续点形成的向量

3. **计算 ego 运动向量**:
   - Ego 向量: 从 $t-1$ 时刻指向 $t$ 时刻
   $$\hat{v}_{ego}^t = p_{ego}^t - p_{ego}^{t-1}$$

4. **角度差损失**:
$$\mathcal{L}_{dir}^t = F_{ang}(\hat{v}_m^t, \hat{v}_{ego}^t)$$

其中 $F_{ang}$ 是角度差函数:
$$F_{ang}(v_1, v_2) = \arccos\left(\frac{v_1 \cdot v_2}{\|v_1\| \|v_2\|}\right)$$

5. **总方向损失**:
$$\mathcal{L}_{dir} = \frac{1}{T_f} \sum_{t=1}^{T_f} \mathcal{L}_{dir}^t$$

**效果**:
- 正则化轨迹方向
- 防止车辆逆行或偏离车道

**与传统方法对比**:

| 方法 | 约束形式 | 计算复杂度 | 可解释性 |
|------|---------|-----------|---------|
| UniAD (Cost Map) | 密集栅格（200×200） | 高 | 中 |
| VAD (Vectorized) | 实例级向量（~100个） | 低 | 高 |

#### 3.5 End-to-End Learning（端到端学习）

##### 3.5.1 Imitation Learning Loss（模仿学习损失）

**目的**: 监督规划轨迹模仿人类专家驾驶行为

$$\mathcal{L}_{imi} = \frac{1}{T_f} \sum_{t=1}^{T_f} \|V_{ego}^t - \hat{V}_{ego}^t\|_1$$

其中:
- $V_{ego}$: Ground truth 自车轨迹（从 nuScenes 标注获取）
- $\hat{V}_{ego}$: 预测的规划轨迹

**为什么用 L1 而非 L2**:
- L1 对异常值（outliers）更鲁棒
- L2 会过度惩罚大误差

##### 3.5.2 Overall Training Objective（总训练目标）

$$\mathcal{L}_{total} = \omega_1 \mathcal{L}_{map} + \omega_2 \mathcal{L}_{motion\_total} + \omega_3 \mathcal{L}_{col} + \omega_4 \mathcal{L}_{bd} + \omega_5 \mathcal{L}_{dir} + \omega_6 \mathcal{L}_{imi}$$

**权重设置**（论文中的超参数）:
- $\omega_1, \omega_2$: 感知和预测损失权重
- $\omega_3, \omega_4, \omega_5$: 约束损失权重（需要平衡，避免过度约束）
- $\omega_6$: 模仿学习权重

**训练策略**:
1. **Two-stage Training**（可选）:
   - Stage 1: 先训练感知和预测模块（冻结规划头）
   - Stage 2: 端到端联合训练

2. **End-to-End Training**（VAD 采用）:
   - 所有模块同时训练
   - 梯度可以从规划损失反向传播到感知模块
   - 允许感知模块学习对规划有用的特征

**优化器配置**:
- AdamW optimizer
- Learning rate: $2 \times 10^{-4}$
- Weight decay: 0.01
- Cosine Annealing scheduler
- 60 epochs on nuScenes

---

### 4. Experiments（实验详解）

#### 4.1 Dataset and Evaluation Protocol

**nuScenes Dataset**:
- **规模**: 1000 个驾驶场景，每个约 20 秒
- **标注**: 1.4M 3D bounding boxes，23 个类别
- **相机配置**: 6 个环视相机，360° FOV
- **采样率**: 2Hz keyframes
- **划分**: 700 train / 150 val / 150 test

**评估指标**:

1. **Displacement Error (L2 Error)**:
$$\text{L2}(t) = \frac{1}{N} \sum_{i=1}^{N} \|p_{ego}^{t,i} - \hat{p}_{ego}^{t,i}\|_2$$
   - 在 1s, 2s, 3s 分别计算
   - 平均值作为总体指标

2. **Collision Rate (CR)**:
$$\text{CR}(t) = \frac{\text{\# of collisions at time } t}{\text{total \# of samples}} \times 100\%$$
   - 碰撞判定: ego bounding box 与 agent bounding box 有 IoU
   - 在 1s, 2s, 3s 分别计算

3. **Inference Speed**:
   - FPS (Frames Per Second)
   - Latency (ms per frame)
   - 测试硬件: NVIDIA RTX 3090 / Tesla A100

#### 4.2 Implementation Details

**两个模型变体**:

| 配置项 | VAD-Tiny | VAD-Base |
|--------|----------|----------|
| Backbone | ResNet50 | ResNet50 |
| 输入尺寸 | 640×360 | 1280×720 |
| BEV Queries | 100×100 | 200×200 |
| BEV Encoder Layers | 3 | 6 |
| Map Queries | 100 | 100 |
| Agent Queries | 300 | 300 |
| Motion Decoder Layers | 3 | 6 |
| Map Decoder Layers | 3 | 6 |
| Hidden Dim | 256 | 256 |
| 参数量 | ~50M | ~80M |

**感知范围**:
- 纵向: 60m（前方）
- 横向: ±30m（左右各 15m）

**时间配置**:
- 历史: 2 秒（4 帧 @ 2Hz）
- 未来: 3 秒（6 帧 @ 2Hz）

**训练配置**:
- 8 × NVIDIA RTX 3090 GPUs
- Batch size: 1 per GPU（梯度累积 8 步）
- 训练时间: ~3 天（60 epochs）

#### 4.3 Main Results（主要结果）

##### 4.3.1 Open-Loop Planning Performance

**与 SOTA 方法对比**:

| Method | L2 (m) Avg. | Collision (%) Avg. | FPS | Latency (ms) |
|--------|-------------|-------------------|-----|--------------|
| NMP (LiDAR) | - | 1.92 | - | - |
| SA-NMP (LiDAR) | - | 1.59 | - | - |
| FF (LiDAR) | 1.43 | 0.43 | - | - |
| EO (LiDAR) | 1.60 | 0.33 | - | - |
| ST-P3 | 2.11 | 0.71 | 1.6 | 628.3 |
| UniAD | **1.03** | 0.31 | 1.8 | 555.6 |
| **VAD-Tiny** | 0.78 | 0.38 | **16.8** | **59.5** |
| **VAD-Base** | **0.72** | **0.22** | **4.5** | **224.3** |

**关键观察**:

1. **精度提升（VAD-Base vs. UniAD）**:
   - L2 误差: 1.03m → 0.72m（**↓30.1%**）
   - 碰撞率: 0.31% → 0.22%（**↓29.0%**）
   - 在所有时间步（1s/2s/3s）均有提升

2. **速度提升**:
   - VAD-Base: **2.5× 加速**（1.8 FPS → 4.5 FPS）
   - VAD-Tiny: **9.3× 加速**（1.8 FPS → 16.8 FPS）
   - 接近实时性能（10+ FPS）

3. **效率-性能权衡（VAD-Tiny）**:
   - 仍优于 ST-P3（0.78m vs. 2.11m）
   - 碰撞率略高于 VAD-Base 但可接受
   - 非常适合计算资源受限场景

4. **与 LiDAR 方法对比**:
   - Vision-only 的 VAD 性能优于多数 LiDAR-based 方法
   - 证明矢量化表示的有效性

**Ego Status 的影响**:

论文还测试了使用 ego status（速度、加速度）作为输入的版本:

| Method | L2 (m) Avg. | Collision (%) Avg. |
|--------|-------------|-------------------|
| VAD-Tiny (w/ ego status) | **0.41** | 0.16 |
| VAD-Base (w/ ego status) | **0.37** | 0.14 |

- 性能大幅提升，但可能存在 shortcut learning
- 主实验不使用以确保评估公平性

##### 4.3.2 Closed-Loop Simulation Results

**CARLA Town05 Benchmark**:

| Method | Town05 Short |  | Town05 Long |  |
|--------|-------------|-------------|-------------|-------------|
|  | DS ↑ | RC ↑ | DS ↑ | RC ↑ |
| CILRS | 7.47 | 13.40 | 3.68 | 7.19 |
| LBC | 30.97 | 55.01 | 7.05 | 32.09 |
| Transfuser (LiDAR) | 54.52 | 78.41 | 33.15 | 56.36 |
| ST-P3 | 55.14 | 86.74 | 11.45 | 83.15 |
| **VAD-Base** | **64.29** | **87.26** | **30.31** | **75.20** |

**指标说明**:
- **DS (Driving Score)**: 综合驾驶质量得分（考虑安全性、舒适性、规则遵守）
- **RC (Route Completion)**: 路径完成率

**关键观察**:
1. VAD 在 **Town05 Short** 上达到最佳 vision-only 性能
   - DS 提升 9.15（相比 ST-P3）
   - RC 基本持平但更安全

2. **Town05 Long** 更具挑战性:
   - VAD DS 接近 LiDAR-based 方法（30.31 vs. 33.15）
   - RC 显著优于 ST-P3（75.20 vs. 83.15）
   - 证明长距离场景中的泛化能力

#### 4.4 Ablation Study（消融实验详解）

##### 4.4.1 Design Choices Ablation

**实验设置**: 逐步移除各设计组件

| ID | Agent Inter. | Map Inter. | Overstep Const. | Dir. Const. | Col. Const. | L2 (m) Avg. | CR (%) Avg. |
|----|--------------|------------|-----------------|-------------|-------------|-------------|-------------|
| 1 | ✓ | ✗ | ✓ | ✓ | ✓ | 0.86 | 0.29 |
| 2 | ✗ | ✓ | ✓ | ✓ | ✓ | 0.82 | 0.26 |
| 3 | ✓ | ✓ | ✗ | ✗ | ✗ | 0.76 | 0.28 |
| 4 | ✓ | ✓ | ✓ | ✗ | ✗ | 0.80 | 0.24 |
| 5 | ✓ | ✓ | ✗ | ✓ | ✗ | 0.75 | 0.25 |
| 6 | ✓ | ✓ | ✗ | ✗ | ✓ | 0.77 | 0.26 |
| 7 | ✓ | ✓ | ✓ | ✓ | ✓ | **0.72** | **0.22** |

**关键发现**:

1. **Ego-Map Interaction 的重要性（ID 1）**:
   - 移除后 L2 误差增加 0.14m（+19.4%）
   - 没有地图信息，难以规划合理轨迹

2. **Ego-Agent Interaction 的重要性（ID 2）**:
   - 移除后碰撞率增加
   - 无法感知其他车辆意图

3. **三个约束的贡献（ID 4-6 vs. ID 7）**:
   - 每个约束单独使用都有帮助
   - **组合使用效果最佳**（ID 7）
   - 证明约束的互补性

4. **无约束 baseline（ID 3）**:
   - 仅靠隐式学习性能较差
   - 显式约束至关重要

##### 4.4.2 Rasterized vs. Vectorized Representation

**对比实验**: VAD 使用栅格化地图 vs. 矢量化地图

| Map Repr. | Dir. Const. | Overstep Const. | L2 (m) Avg. | CR (%) Avg. |
|-----------|-------------|-----------------|-------------|-------------|
| Rasterized | ✗ | ✗ | 0.74 | **0.39** |
| Vectorized | ✗ | ✗ | 0.77 | 0.26 |
| Vectorized | ✓ | ✓ | **0.72** | **0.22** |

**实现细节**:
- Rasterized 版本: 使用 map queries 预测 BEV 语义分割（3 类）
- 分割图分辨率: 200×200

**关键观察**:
1. **矢量化在约束前性能相近**（0.74 vs. 0.77）
2. **矢量化支持显式约束**:
   - 可以计算到具体边界/车道的距离
   - 栅格化难以施加实例级约束
3. **碰撞率差异显著**（0.39% vs. 0.22%）:
   - 矢量化提供更精确的结构信息

##### 4.4.3 Module Runtime Analysis

**VAD-Tiny 各模块耗时**（on RTX 3090）:

| Module | Latency (ms) | Proportion |
|--------|--------------|-----------|
| Backbone | 23.2 | 39.0% |
| BEV Encoder | 12.3 | 20.7% |
| Motion Module | 11.5 | 19.3% |
| Map Module | 9.1 | 15.3% |
| Planning Module | **3.4** | **5.7%** |
| **Total** | **59.5** | **100.0%** |

**关键洞察**:
1. **规划模块非常高效**（仅 3.4ms）:
   - 得益于稀疏矢量化表示
   - 简洁的 MLP-based decoder

2. **特征提取占主导**:
   - Backbone + BEV Encoder: 59.7%
   - 优化空间: 使用更轻量 backbone（如 EfficientNet）

3. **感知模块仍需优化**:
   - Motion + Map: 34.6%
   - 但相比密集预测已大幅降低

#### 4.5 Qualitative Results（定性结果分析）

**可视化示例**（论文 Figure 4）:

**场景 1: 直行场景**
- 检测: 3 辆前方车辆，2 条车道线
- 运动预测: 所有车辆直行（单一模态高置信度）
- 规划: 沿车道中心直行，保持与前车安全距离

**场景 2: 交叉口左转**
- 检测: 对向来车、左侧车道线
- 运动预测: 对向车直行（碰撞风险）
- 规划: 等待对向车通过后左转（体现 collision constraint 作用）

**场景 3: 复杂多车场景**
- 检测: 5+ 车辆，多条车道线和边界
- 运动预测: 多模态（部分车可能变道）
- 规划: 选择最保守路径，避开所有潜在碰撞

**矢量化表示的优势**:
- 清晰的结构信息可视化
- 每个元素都有明确语义（哪条是边界、哪条是车道）
- 便于调试和验证

---

### 5. Conclusion & Discussion（结论与讨论）

#### 5.1 主要贡献总结

1. **范式转变**: 
   - 从栅格化表示到矢量化表示
   - 证明了矢量化在端到端自动驾驶中的可行性和优越性

2. **技术创新**:
   - Query interaction 机制实现高效信息融合
   - 三个实例级矢量化约束提升安全性
   - 隐式 + 显式双重利用场景信息

3. **性能突破**:
   - SOTA 规划精度（L2 0.72m）
   - 最低碰撞率（0.22%）
   - 高推理速度（4.5 FPS for Base, 16.8 FPS for Tiny）

#### 5.2 Limitations（局限性）

1. **开环评估的局限**:
   - nuScenes 是开环数据集，无法评估交互式驾驶
   - 真实部署中的闭环反馈可能带来新挑战

2. **多模态预测的利用不足**:
   - 当前仅使用最高置信度模态
   - 未来可探索如何利用多模态进行风险感知规划

3. **对在线建图的依赖**:
   - 感知模块失败会直接影响规划
   - 复杂/不规则场景（如施工区域）可能难以矢量化

4. **确定性规划**:
   - 输出单一轨迹，无不确定性量化
   - 后续 VADv2 引入概率规划解决此问题

5. **场景泛化性**:
   - 仅在 nuScenes（城市道路）和 CARLA 评估
   - 高速公路、乡村道路等场景未验证

#### 5.3 Future Directions（未来方向）

1. **概率规划**:
   - 输出轨迹分布而非单一轨迹
   - 量化规划的不确定性（→ VADv2）

2. **更丰富的交通信息**:
   - 交通标志（限速、禁停）
   - 交通灯状态
   - 车道图拓扑（lane connectivity）

3. **闭环评估与真实部署**:
   - 在真实车辆上测试
   - 长时域稳定性验证

4. **多模态输入融合**:
   - 结合 LiDAR 提升感知鲁棒性
   - 融合 GPS/IMU 提升定位精度

5. **更高效的架构**:
   - 知识蒸馏（从 VAD-Base 到更小模型）
   - 模型量化（INT8）
   - 达到 30+ FPS 以满足实时性

---

## 整体总结

### 核心思想与创新

**核心思想**: 用矢量化场景表示替代栅格化表示，实现高效且安全的端到端自动驾驶

**三层创新**:

1. **表示层（Representation）**:
   - 矢量化地图（lane vectors, boundary vectors）
   - 矢量化运动（multi-modal agent trajectories）
   - 稀疏、结构化、高效

2. **架构层（Architecture）**:
   - Query-based 感知（借鉴 DETR 系列）
   - Query interaction 机制（ego-agent, ego-map）
   - 端到端可微分训练

3. **约束层（Constraints）**:
   - 实例级几何约束
   - 隐式学习 + 显式约束
   - 动态安全 + 静态安全 + 合理性

### 技术亮点深度解析

#### 1. 为什么矢量化比栅格化好？

**计算复杂度对比**:

| 操作 | 栅格化（200×200） | 矢量化（~100 elements） |
|------|------------------|----------------------|
| 碰撞检测 | O(40000) 像素遍历 | O(100) 向量距离计算 |
| 边界检测 | 密集分割 + 后处理 | 直接点线距离 |
| 方向约束 | 难以施加 | 直接角度计算 |

**信息保真度**:
- 栅格化: 离散化损失，分辨率受限
- 矢量化: 连续坐标，精确到厘米级

#### 2. Query Interaction 的直觉理解

可类比为 "注意力会议":
- **Ego Query**: 决策者（自车）
- **Agent Queries**: 汇报其他车辆状态的顾问
- **Map Queries**: 汇报道路状况的顾问
- **Cross-Attention**: 决策者向顾问提问的机制

决策者通过提问（query）获取所需信息（key-value），而非被动接收所有信息。

#### 3. 三个约束的互补性

| 约束 | 保护对象 | 约束类型 | 时间敏感性 |
|------|---------|---------|-----------|
| Collision | 动态安全 | 软约束 | 高（未来轨迹） |
| Boundary | 静态安全 | 软约束 | 中（固定边界） |
| Direction | 合理性 | 正则化 | 低（交通规则） |

组合使用形成多层安全网。

### 与相关工作的对比

#### VAD vs. UniAD

| 维度 | UniAD | VAD |
|------|-------|-----|
| **表示** | 密集（BEV 栅格图） | 稀疏（矢量） |
| **中间任务** | 跟踪、占用预测 | 矢量化地图、运动预测 |
| **规划方式** | Goal-oriented query | Query interaction + 约束 |
| **后处理** | 需要（轨迹采样） | 不需要 |
| **可解释性** | 中（有中间表示） | 高（矢量可视化） |
| **效率** | 低（1.8 FPS） | 高（4.5 FPS） |

#### VAD vs. ST-P3

| 维度 | ST-P3 | VAD |
|------|-------|-----|
| **Cost Map** | 需要 | 不需要 |
| **Planning** | 基于 cost 采样 | 直接回归 |
| **速度** | 1.6 FPS | 4.5 FPS |
| **L2 误差** | 2.11m | 0.72m |

### 对自动驾驶领域的影响

1. **研究方向**:
   - 证明矢量化范式的可行性
   - 启发后续工作（VADv2, SparseDrive, UniV3D）
   - 推动端到端自动驾驶向实用化迈进

2. **工业价值**:
   - 高效率适合车载部署
   - 可解释性满足安全审核需求
   - 模块化设计便于工程化

3. **理论贡献**:
   - 显式约束 + 隐式学习的结合
   - Query-based 统一框架的应用
   - 矢量化表示的系统化研究

### 进一步思考

**开放问题**:

1. **矢量化的边界在哪**？
   - 非结构化场景（越野、停车场）如何矢量化？
   - 极端天气下感知失败怎么办？

2. **如何利用多模态**？
   - 当前仅用最佳模态，浪费了其他模态信息
   - 能否用于风险评估？

3. **如何处理长尾场景**？
   - 施工区域、临时障碍物
   - 矢量化可能不适用，需要混合表示？

**与 VADv2 的联系**:
- VAD: 确定性规划，单一轨迹
- VADv2: 概率规划，多假设轨迹，鲁棒性更强
- VAD 是基础，VADv2 是演进

---

## 关键术语表（Glossary）

### 场景表示相关
- **Rasterized Representation**: 栅格化表示，将场景离散化为像素网格
- **Vectorized Representation**: 矢量化表示，将场景表示为连续坐标的点、线集合
- **BEV (Bird's Eye View)**: 鸟瞰图，俯视角度的场景表示
- **Occupancy Map**: 占用地图，指示空间中哪些区域被占据
- **Cost Map**: 代价地图，每个位置的规划代价（用于路径搜索）

### 感知相关
- **Query**: 可学习的嵌入向量，用于从特征图中提取信息
- **Deformable Attention**: 可变形注意力，允许注意力采样点偏移
- **Permutation Invariance**: 排列不变性，输出不依赖输入顺序
- **Hungarian Matching**: 匈牙利匹配，用于预测和 GT 的最优二分匹配

### 运动预测相关
- **Multi-Modality**: 多模态，预测多种可能的未来轨迹
- **Waypoint**: 航点，轨迹上的离散点
- **Min-of-N Loss**: 仅对最佳预测计算损失，鼓励多样性
- **Social Interaction**: 社交交互，agent 之间的相互影响

### 规划相关
- **Imitation Learning**: 模仿学习，学习模仿专家行为
- **Ego Vehicle**: 自车，被控制的车辆
- **Driving Command**: 驾驶指令，高层导航意图（左转/右转/直行）
- **Open-Loop**: 开环评估，不与环境交互，仅评估单步预测
- **Closed-Loop**: 闭环评估，持续交互，评估长时域性能

### 评估指标
- **Displacement Error (L2)**: 位移误差，预测位置与真实位置的欧氏距离
- **Collision Rate**: 碰撞率，发生碰撞的样本占比
- **FPS (Frames Per Second)**: 每秒处理帧数，衡量推理速度
- **Driving Score (DS)**: 驾驶得分，CARLA 中的综合评估指标
- **Route Completion (RC)**: 路径完成率，成功到达目的地的比例

---

## 论文要点速查

**一句话总结**: VAD 用矢量化场景表示替代栅格化表示，通过 query interaction 和三个实例级约束实现高效、安全的端到端自动驾驶。

**三大优势**: 
1. 计算效率高（2.5-9.3× 加速）
2. 结构信息完整（实例级约束）
3. 性能优异（SOTA 精度和安全性）

**关键技术**:
- Vectorized Map + Motion Prediction
- Ego-Agent & Ego-Map Query Interaction
- 三个矢量化规划约束（碰撞/边界/方向）

**主要结果**:
- L2: 0.72m（↓30.1% vs. UniAD）
- 碰撞率: 0.22%（↓29.0% vs. UniAD）
- 速度: 4.5 FPS（↑2.5× vs. UniAD）

**适用场景**: 城市道路端到端自动驾驶，尤其是计算资源受限的车载部署

**局限性**: 开环评估、确定性规划、依赖在线建图质量




