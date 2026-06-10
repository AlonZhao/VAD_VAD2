# VADv2 论文精读（ICLR 2026 版）+ 代码对照

> **来源**：`VADv2 End-to-End Vectorized Autonomous Driving via Probabilistic Planning.pdf`
>
> **⚠️ 重要澄清**：本 PDF 是 **ICLR 2026 会议版**，主战场是 **CARLA 闭环**（不是 2024 arXiv 那篇 nuScenes 开环版）。本仓库的 `VADv2/` 代码与这篇论文配套。
>
> **本文做什么**：把论文方法（带公式）和仓库代码逐一对照，确保理解准确、可追溯。

---

## 0. 一句话定位

> VADv2 把规划建模成**场景条件下的概率分布 `p(a|o)`**：离散化动作空间成轨迹词表，用大规模驾驶示范学每条轨迹的概率，推理时采样一条执行。灵感来自 **GPT**——下一个词非确定性，LLM 学的是上下文条件下的概率分布。

---

## 1. 整体架构（论文 Fig.2 + §3.1）

```
多视角图像序列
    │ Encode
    ↓
┌─────────────────────────────────┐
│ Scene Encoder → 4类 scene tokens │
│   M: Map tokens(车道/边界/人行道) │  ← MapTR
│   A: Agent tokens(他车+未来轨迹)  │  ← VAD
│   T: Traffic Element tokens       │  ← CARLA特有(红绿灯+停止标志)
│   I: Image tokens(前视密集特征)   │  ← 补充instance token
└─────────────────────────────────┘
    │ 作为 K,V              + E_navi(导航) + E_state(自车状态)
    ↓
┌─────────────────────────────────┐
│ Planning Transformer (cascaded)  │
│   Q = 词表轨迹的编码 E(a)         │
│   K,V = scene tokens              │
└─────────────────────────────────┘
    ↓
概率分布 p(a)  →  Conflict Check(场景约束) → 采样一条执行
    ↑
监督: 大规模驾驶示范(KL/交叉熵) + 场景约束(conflict loss)
```

**代码对照**：4 类 token 对应 `VADv2_head.py` 里的 detection/map head + `tl_*_cls_branch`（traffic element）+ 前视特征。

---

## 2. 核心：Probabilistic Planning（§3.2）

### 2.1 建模为概率分布（公式1）

```
p(a|o),  o = (E_scene, E_navi, E_state),  a = (x₁,y₁, x₂,y₂, ..., x_T,y_T)
```
- `o`：场景观测（scene tokens + 导航 + 自车状态）
- `a`：未来 T 个路点的轨迹
- 把"规划"变成"在给定场景下,每条轨迹的概率"

> **规划视角**：这就是给 lattice 每条候选轨迹算一个"概率"而非"cost"。本质同构,只是从能量/代价空间换到概率空间。

### 2.2 轨迹词表生成：Furthest Trajectory Sampling（Algorithm 1）⭐

**这里修正一个常见误解**：词表**不是 k-means 聚类**得来的,而是**最远轨迹采样**(furthest trajectory sampling)!

论文 Algorithm 1 逻辑：
```
输入: 所有示范轨迹集合 S, 词表大小 N=4096
输出: 词表 V

V ← ∅
for i in 1..N-1:
    if V 空:
        随机选一条 a 放进 V
    max_dis = 0
    for S 中每条轨迹 a:
        dis = calculate_distance(a, V)   # a的终点到V中所有轨迹终点的最小距离
        if dis > max_dis:
            max_dis = dis
            â = a                          # 记录"离V最远"的那条
    V ← V ∪ {â}                            # 把最远的加入词表
return V
```

**核心思想**:每次都挑"离已选集合最远"的轨迹加入,保证词表**覆盖最广、最分散**。

> **为什么不用 k-means**:论文对比了 Trajeglish 用 k-disk sampling、MotionLM 用 axis-aligned 量化。furthest sampling 的好处是**保证覆盖度**(不会因为某类轨迹样本多就过度密集),且每条都是真实示范轨迹→**天然满足运动学约束**(转成 steer/throttle/brake 不超限)。

**代码对照**(VADv2_head.py:204):
```python
self.plan_anchors = np.load('carla_plan_vocabulary_4096.npy')  # 离线采样好的词表
```
词表是离线用 Algorithm 1 算好存成 .npy,代码直接加载。

### 2.3 轨迹编码：NeRF 风格高频编码（公式2）

```
E(a) = (Γ(xᵢ), Γ(yᵢ))ᵢ₌₁..T
Γ(pos) = (γ(pos,j))ⱼ₌₀..L-1
γ(pos,j) = (cos(pos/10000^{2πj/L}), sin(pos/10000^{2πj/L}))
```
- 每个坐标值映射到高维空间 ℝ^{2L}
- **灵感来自 NeRF**:用高频位置编码逼近高频函数,让网络能区分相近的轨迹
- 类似 Transformer 的正弦位置编码,但用在轨迹坐标上

### 2.4 打分（公式3）

```
p(a) = σ(MLP(φ(E(a), E_scene) + E_navi + E_state))
```
- `φ`:cascaded Transformer decoder,`E(a)`作 query,`E_scene`作 K,V
- 加上导航 `E_navi` 和自车状态 `E_state`
- `σ`:sigmoid → 每条轨迹一个概率

**代码对照**:`plan_*_branch` 系列分支 + Transformer decoder。

---

## 3. 训练：三种监督（§3.3）

### 3.1 Distribution Loss（公式4→5）核心

```
L_distribution = D_KL(p_data || p_pred)             (4) KL散度
               = -Σ p_data(a)·log p_pred(a)         (5) 等价于交叉熵
```

**怎么得到 p_data**:
- 对每帧,从词表里选**离真值轨迹 L2 最近**的那条,标签设 1,其余 0
- 统计所有帧,每条轨迹被选中的**频率**就是 `p_data(a)`

> **和 LLM 完全同构**:LLM 把真值 token 标 1、其余标 0,用交叉熵学下一词分布。VADv2 把"最匹配的轨迹"标 1,学下一条轨迹的分布。**轨迹 = 词,词表 = 词典**。

### 3.2 Conflict Loss（公式6）

```
L_conflict = Σ 𝟙_conflict(a)·log p_pred(a)
```
- `𝟙_conflict(a)`:轨迹 a 若和**他车真值未来轨迹**或**道路边界**冲突 → =1
- 把冲突轨迹当**负样本**,压低其概率

> **规划视角**:这是把"碰撞检测/边界检测"作为**软的负样本监督**注入训练。注意——仍是软的,降低概率而非硬禁止。**量产仍需硬约束兜底**。

**代码对照**:`plan_cls_col_branch`(碰撞)、`plan_cls_bd_branch`(边界)等打分分支。

### 3.3 Scene Token Loss

各 token 用对应监督,保证显式编码高层信息:
- **Map tokens**:同 MapTRv2(L1 点回归 + Focal 分类)
- **Agent tokens**:检测 loss + motion 预测 loss(同 VAD,minFDE 选代表轨迹算 L1)
- **Traffic element**:红绿灯/停止标志分类

---

## 4. v1 → v2 完整对比（结合论文）

| 维度 | VAD v1 | VADv2(ICLR2026) |
|---|---|---|
| **规划范式** | 回归一条轨迹 | 概率分布 p(a\|o),词表采样 |
| **理论灵感** | 轨迹优化 | **GPT/LLM 的下一词预测** |
| **输出** | 3条(左/直/右) | 4096条概率分布 |
| **词表生成** | 无 | **furthest trajectory sampling**(非k-means) |
| **轨迹编码** | — | NeRF风格高频位置编码 |
| **主监督** | L1回归 + 约束loss | **KL/交叉熵**(distribution loss) |
| **约束** | 训练软loss | conflict loss(负样本)+ 显式打分 |
| **评测主场** | nuScenes开环 | **CARLA闭环**(DS/RC)+ NAVSIM(PDMS) |
| **额外输入** | — | 红绿灯/停止标志(traffic element) |
| **多模态** | 弱(易平均) | 强(概率多峰) |

---

## 5. 论文里值得记住的几个点（面试加分）

### 5.1 为什么概率建模？(动机)
论文明说:**借鉴 GPT**。给定上下文,下一个词是非确定的,LLM 学的是"上下文条件下的概率分布"再采样。驾驶同理——同一场景下,合理的开法不止一种(变道/跟车都对),确定性回归强行学一条会平均化。概率建模天然处理这种**多模态不确定性**。

### 5.2 单步 vs 迭代(对比 MotionLM)
论文 §2 点评:MotionLM 把轨迹拆成单步 action token 迭代 rollout,会**误差累积**且可能违反物理约束。**VADv2 每个 action token 是一条完整轨迹**,one-shot 出解,不累积误差、保证运动学可行。

> 这点很重要——VADv2 的"词"是整条轨迹,不是单步动作。这是它和 MotionLM/Trajeglish 这类自回归方案的本质区别。

### 5.3 furthest sampling 保证运动学可行
词表每条都来自真实示范 → 转成 steer/throttle/brake 不超限。比解析式撒点(可能撒出不可行轨迹)更安全。

### 5.4 闭环才是主战场
ICLR2026 版重点是 CARLA 闭环 DS(Driving Score)/RC(Route Completion),不是 nuScenes 开环 L2。论文也对比了 RAD、DiffusionDrive、GoalFlow、Hydra-MDP++ 等 2025 SOTA。

---

## 6. 规划视角总结（你的话术）

> "VADv2 这篇 ICLR2026 版,核心是把规划建模成**概率分布**,思路直接借鉴 GPT——同一场景合理开法不止一种,确定性回归会平均化,概率建模天然多模态。
>
> 它的轨迹词表我特别有共鸣:用 **furthest sampling** 从真实示范里挑 4096 条最分散的轨迹——这比我们规则 lattice 的解析式撒点更聪明,因为每条都是真人开过的,**天然满足运动学约束**,不会撒出不可行轨迹。本质就是个'数据驱动的、保证覆盖度的 lattice'。
>
> 训练用 KL/交叉熵,和 LLM 学下一词完全同构(最匹配轨迹标1)。约束方面有个 conflict loss 把碰撞/越界轨迹当负样本压低概率——但注意这仍是软的,降低概率不是硬禁止,**量产我还是会加硬碰撞检测兜底**。
>
> 而且这版重点是 CARLA **闭环**评测,比 v1 的 nuScenes 开环更接近真实——这点很关键,因为开环 L2 低不代表闭环能开。"

---

## 7. 修正之前文档的说法

之前 `VAD规划模块详解.md` 和总结里说"本仓库 VADv2 是 CARLA 适配版,非原版"——**措辞要修正**:

- ❌ 旧说法:"本仓库是 CARLA 适配版,原版是 nuScenes"
- ✅ 准确说法:存在**两个 VADv2 版本**——2024 arXiv 版(nuScenes 开环为主)和 **ICLR2026 版(CARLA 闭环为主,本仓库+本PDF配套)**。词表生成是 furthest sampling 不是 k-means。

---

## 8. 三句话记住

1. **概率规划**:p(a|o),借鉴 GPT,词表轨迹=词,KL/交叉熵训练,天然多模态。
2. **词表 = furthest sampling 的 lattice**:4096条真实示范轨迹,最分散,保运动学可行。
3. **ICLR2026 版主打 CARLA 闭环**,conflict loss 软约束仍需硬兜底。
