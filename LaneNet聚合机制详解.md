# LaneNet 聚合机制详解

> **作用**：把一条车道线的 20 个点特征聚合成 1 个向量表示
>
> **代码位置**：`VAD_head.py:40-69`

---

## 核心思想

**PointNet 风格的迭代 max-pooling**：通过多层"逐点 MLP → 全局 max → 拼接"逐步提取全局特征。

---

## 结构

```python
LaneNet(in_channels=256, hidden_unit=128, num_subgraph_layers=3)
```

**3 层结构**：
- Layer 0: `MLP(256 → 128)` + max-pool + concat
- Layer 1: `MLP(256 → 128)` + max-pool + concat （输入是上层拼接后的256维）
- Layer 2: `MLP(256 → 128)` + max-pool + concat
- 最终输出: 再做一次 max → `(256,)` 一条线的向量表示

**MLP 定义**：
```python
nn.Sequential(
    nn.Linear(in_channels, hidden_unit),
    nn.LayerNorm(hidden_unit),
    nn.ReLU()
)
```

---

## 前向传播流程

**输入**：`(B, 100, 20, 256)`
- B = batch size
- 100 = 车道线数量
- 20 = 每条线的点数
- 256 = 每个点的特征维度

**逐层处理**（以一条线为例）：

```python
x = pts_lane_feats  # (B, 100, 20, 256)

# ─── Layer 0 ───
x = MLP_0(x)                              # (B, 100, 20, 256) → (B, 100, 20, 128)
x_max = torch.max(x, dim=-2)[0]           # (B, 100, 128) 在20个点上取max
x_max = x_max.unsqueeze(2).repeat(..., 20, ...)  # (B, 100, 20, 128) 广播回每个点
x = torch.cat([x, x_max], dim=-1)         # (B, 100, 20, 256) 拼接局部+全局

# ─── Layer 1 ───
x = MLP_1(x)                              # (B, 100, 20, 256) → (B, 100, 20, 128)
x_max = torch.max(x, dim=-2)[0]           # (B, 100, 128)
x_max = x_max.unsqueeze(2).repeat(..., 20, ...)
x = torch.cat([x, x_max], dim=-1)         # (B, 100, 20, 256)

# ─── Layer 2 ───
x = MLP_2(x)                              # (B, 100, 20, 256) → (B, 100, 20, 128)
x_max = torch.max(x, dim=-2)[0]           # (B, 100, 128)
x_max = x_max.unsqueeze(2).repeat(..., 20, ...)
x = torch.cat([x, x_max], dim=-1)         # (B, 100, 20, 256)

# ─── 最终输出 ───
x_max = torch.max(x, dim=-2)[0]           # (B, 100, 256) 最后一次max
return x_max
```

**输出**：`(B, 100, 256)` 每条线一个 256 维向量

---

## 三个关键机制

### 1. 逐点 MLP（局部特征提取）
```python
x = MLP(x)  # 对每个点独立做 Linear + Norm + ReLU
```
让每个点学习"我在这条线上的局部特征"（例如：弯道点 vs 直道点）。

### 2. Max-pooling（全局特征聚合）
```python
x_max = torch.max(x, dim=-2)[0]  # 在20个点上取最大值
```
- **排列不变**：点序 A→B 或 B→A 结果相同（适合车道线方向标注不一致）
- **突出关键特征**：保留"最显著的点"（例如最大曲率点）

### 3. 拼接全局信息（全局-局部融合）
```python
x = torch.cat([x, x_max], dim=-1)  # 局部特征(128) + 全局特征(128) = 256
```
让下一层 MLP 基于"整条线的上下文"进一步提炼特征。

---

## 可视化流程

```
输入: 一条车道线 20个点 (20, 256)
    ↓
Layer 0: 每个点MLP(256→128) → Max得全局(128) → 拼接(20,256)
         每个点现在知道"整条线的全局信息"
    ↓
Layer 1: MLP(256→128) → Max(128) → 拼接(20,256)
         全局信息更抽象
    ↓
Layer 2: MLP(256→128) → Max(128) → 拼接(20,256)
         最终全局特征
    ↓
最终Max: (20,256) → (256)
    ↓
输出: (256) 编码了"这条线的类别+形状+朝向"
```

---

## 为什么不用简单平均？

| 方案 | 优势 | 劣势 |
|---|---|---|
| 简单平均 `mean(x)` | 简单 | ❌ 丢失关键点信息<br>❌ 无法学习非线性关系 |
| **LaneNet（PointNet风格）** | ✅ 学习非线性关系<br>✅ 排列不变<br>✅ 关键特征突出<br>✅ 全局-局部融合 | 参数稍多（~100K） |

---

## 特征含义

经过 LaneNet 后，每条车道线的 256 维向量包含：
- **类别信息**：分道线 / 边界线 / 人行道
- **几何信息**：直线 / 曲线 / 曲率
- **朝向信息**：线的走向（用于方向约束）
- **上下文信息**：和其他线的相对位置

这个向量会喂给 `ego_map_decoder`，让 ego 学习"关注哪些车道线相关"（当前车道 + 要拐入的车道）。

---

## 类比理解

**LaneNet = 分析一条推文的情感**

```
输入: 20个词的推文（每个词有词向量）
Layer 0: 每个词过MLP → max找"最情绪化的词" → 广播给所有词
Layer 1: 每个词在知道"整体情绪"后再过MLP → 再次max
Layer 2: 重复
输出: 整条推文的情感向量（256维）
```

**车道线也一样**：
- 20 个点 = 20 个词
- Max-pooling = 找"最有代表性的点"
- 拼接 = 让每个点知道"整条线的特性"

---

## 参数量

```python
Layer 0: Linear(256→128) + LayerNorm(128) ≈ 33K
Layer 1: Linear(256→128) + LayerNorm(128) ≈ 33K
Layer 2: Linear(256→128) + LayerNorm(128) ≈ 33K
总计: 约 100K 参数（很轻量）
```

---

## 总结

**一句话**：LaneNet 用 **3 层"逐点 MLP + max-pooling + 拼接"** 逐步提取车道线的全局特征。

**核心思想**：借鉴 PointNet 的排列不变性 + 全局-局部信息融合，适合处理"点集"类数据（车道线、点云）。

**为什么有效**：Max-pooling 保留关键特征（最显著的点），多层迭代让特征从局部→全局→抽象。
