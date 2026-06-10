# VAD 规划模块详解

> **文档定位**：聚焦 VAD 的 Planning Head——端到端规划的核心，也是 VAD 区别于 BEVFormer 的真正原创部分。
>
> **适用对象**：规划方向工程师，理解 ego query 如何与感知/预测结果交互、输出自车轨迹。
>
> **前置知识**：已理解 BEV Encoder（感知主干，复用 BEVFormer）、Detection/Map/Motion Head 的输出。本文从"感知给规划准备了什么"讲起。

---

## 📑 目录

1. [规划开始前：感知/预测给了什么信息](#1-规划开始前感知预测给了什么信息)
2. [Planning Head 代码精读](#2-planning-head-代码精读)（待续）

---

## 0. 为什么单独讲规划

VAD 的感知主干（图像→BEV→检测）基本是 **BEVFormer** 的复用，地图部分借鉴 **MapTR**。VAD 论文真正的原创贡献集中在：

- **向量化的端到端规划**：用 query 显式表示地图、他车、自车
- **ego query 与环境的 cross-attention 交互**
- **三个规划约束损失**（碰撞 / 边界 / 方向）

所以作为规划工程师，这一章才是面试主战场。

---

## 1. 规划开始前：感知/预测给了什么信息

Planning 不是凭空规划的，它消费的是上游 Detection / Map / Motion Head 算好的结果。进 Planning 代码前，先把"桌上摆了哪些菜"理清楚。

### 1.1 食材清单（代码引用）

Planning 段代码（`VAD_head.py:754-807`）开头引用的变量，全部来自上游：

```python
agent_conf  = outputs_classes[-1]            # 来自 Detection Head
agent_query = motion_hs.reshape(...)         # 来自 Motion Head
agent_pos   = outputs_coords_bev[-1]         # 来自 Detection Head
map_query   = self.lane_encoder(map_hs[-1])  # 来自 Map Head
map_conf    = map_outputs_classes[-1]        # 来自 Map Head
map_pos     = map_outputs_coords_bev[-1]     # 来自 Map Head
ego_his_feats = ...                          # 来自自车历史输入（非感知产物）
```

按来源分三类 + 自车输入。

### 1.2 来自 Detection Head（"周围有哪些车，在哪"）

| 变量 | 形状 | 含义 |
|---|---|---|
| `outputs_classes[-1]` → `agent_conf` | (B, 300, 10) | 300个槽位各自的10类置信度 |
| `outputs_coords_bev[-1]` → `agent_pos` | (B, 300, 2) | 300个槽位在BEV上的(x,y)位置 |

**关键细节**：`agent_pos` 用的是 `outputs_coords_bev`，不是完整的 `outputs_coords`。见 `VAD_head.py:627`：

```python
outputs_coords_bev.append(tmp[..., 0:2].clone().detach())  # 归一化BEV坐标，且detach
```

- 只取 `[0:2]`——只要 **xy 平面位置**（规划不关心高度z、尺寸wlh）
- `.detach()`——位置作为"锚点"喂给 planning，梯度不回传到 detection

**Planning 怎么用**：`agent_conf` 筛掉低置信度假目标（`select_and_pad_query`）；`agent_pos` 作为位置编码，告诉 ego"这辆车在你哪个方位"。

### 1.3 来自 Motion Head（"这些车接下来要往哪开"）⭐ 最关键

| 变量 | 形状 | 含义 |
|---|---|---|
| `motion_hs` | (B, 300, 6, 2×256) | 每个agent × 6模态 × 融合特征 |
| → `agent_query`（reshape+MLP后） | (B, 300, 256) | 每个agent压缩成一个特征向量 |

`motion_hs` 是 Motion Head 的最终产物，**已经融合了多样信息**（`VAD_head.py:735`）：

```python
motion_hs = torch.cat([motion_hs, ca_motion_query], dim=-1)
#                      ↑ agent间self-attn  ↑ agent↔map cross-attn
```

喂给 Planning 的 `agent_query` 里，每辆车的特征已经包含：
1. 这辆车自己的检测特征（来自 detection query）
2. 它和周围车的交互（motion_decoder 的 self-attention）
3. 它和地图的关系（motion_map_decoder，"前方有路口→可能转弯"）
4. 它的 6 条多模态未来轨迹意图

**关键转换**（`VAD_head.py:765-766`）：

```python
agent_query = motion_hs.reshape(batch, num_agent, -1)   # (B, 300, 6*2*256=3072)
agent_query = self.agent_fus_mlp(agent_query)           # (B, 300, 256) 6模态融成1个
```

6 个模态的轨迹意图被 `agent_fus_mlp` 压缩进一个 256 维向量。**Planning 拿到的不是"车的位置"，而是"车的位置+意图"的浓缩表示**——这是端到端的核心优势：ego 直接感知到"那辆车想变道"，而非只看到一个静止框。

Motion 还单独输出显式轨迹（给 loss 用，不直接进 ego 交互）：
```python
outputs_traj:       (B, 300, 6, 6, 2)  # 300车 × 6模态 × 6步 × xy
outputs_traj_class: (B, 300, 6)         # 每个模态的概率
```

### 1.4 来自 Map Head（"路在哪，车道怎么走"）

| 变量 | 形状 | 含义 |
|---|---|---|
| `map_hs[-1]` → `map_query`（LaneNet后） | (B, 100, 256) | 100条车道线，每条压成1个向量 |
| `map_outputs_classes[-1]` → `map_conf` | (B, 100, 3) | 每条线的类别置信度 |
| `map_outputs_coords_bev[-1]` → `map_pos` | (B, 100, 20, 2) | 100条线 × 20点 × xy |

与 agent 不同的处理（`VAD_head.py:785-786`）：

```python
map_query = map_hs[-1].view(B, 100, 20, -1)  # 100条线，每条20个点
map_query = self.lane_encoder(map_query)      # LaneNet: (B,100,20,256) → (B,100,256)
```

**LaneNet** 把"一条线的20个点特征"聚合成"这条线的一个特征向量"（类似 PointNet 的 max-pool）。对 ego 来说关心的是"这条车道整体在哪、什么走向"，不需要逐点交互。

**位置用最近点代表**（`VAD_head.py:790-795`）：

```python
map_dis = sqrt(map_pos[...,0]² + map_pos[...,1]²)  # 每个点到自车的距离
min_map_pos_idx = map_dis.argmin(-1)                # 找最近的那个点
min_map_pos = ...                                    # 用它代表整条线的位置
```

一条车道线可能很长，用"离自车最近的点"作为位置编码，因为最近的部分对当下决策最相关。

### 1.5 来自自车输入（非感知产物，但同桌）

| 变量 | 形状 | 含义 | 来源 |
|---|---|---|---|
| `ego_his_trajs` | (B, 1, 2, 3) | 过去1秒2步历史轨迹 | 数据输入 |
| `ego_lcf_feat` | (B, 1, 9) | CAN：速度/加速度/转向等 | 数据输入 |

这两个是**部署时车上直接有的**（不需要预测），可直接喂。`ego_his_trajs` 经 `ego_his_encoder`(LaneNet) 编码成 ego query 的初始特征。

### 1.6 全景图

```
                          bev_embed (B,10000,256)
                                  │
              ┌───────────────────┼───────────────────┐
              ↓                   ↓                   ↓
        Detection Head        Map Head          (Motion 复用Det)
              │                   │                   │
    ┌─────────┴────────┐    ┌─────┴──────┐            │
    ↓                  ↓    ↓            ↓            ↓
agent_conf        agent_pos  map_conf  map_pos    motion_hs
(B,300,10)        (B,300,2) (B,100,3)(B,100,20,2)  │含①检测②车间交互
 "是不是车"        "车在哪"  "是不是路" "路在哪"     │  ③车路交互④意图
    │                  │    │            │            ↓
    │                  │    │            │      agent_query(B,300,256)
    │                  │    │            │       "车在哪+想干嘛"
    └──────┬───────────┘    └─────┬──────┘            │
           ↓                      ↓                   │
    ┌──────────────────────────────────────────────────────┐
    │             ★ Planning Head 的输入桌面 ★              │
    │                                                       │
    │  agent: query(意图) + pos(位置) + conf(筛选)          │
    │  map:   query(车道) + pos(位置) + conf(筛选)          │
    │  ego:   his_trajs(历史) + lcf_feat(CAN)               │
    └──────────────────────────────────────────────────────┘
                            ↓
                  ego query 开始和这些"食材"交互
                  (ego↔agent, ego↔map cross-attention)
```

### 1.7 三个要记住的点（面试高频）

1. **Planning 吃的全是"预测值"不是 GT**——`agent_query` 是 Motion 预测的，`map_query` 是 Map 预测的。训练时故意用带误差的预测，让 ego 学会"在不完美感知下也能规划"，保证训练推理一致。

2. **位置信息都 detach 了**——`outputs_coords_bev` 和 `map_outputs_coords_bev` 都 `.clone().detach()`。位置当锚点用，梯度不回传，避免 planning 的 loss 把检测位置带跑偏。

3. **agent_query 是"位置+意图"的浓缩**——端到端 vs 传统的核心差异。传统规划拿到"框+预测轨迹"两个分离的东西；VAD 的 ego 直接感知一个融合了运动意图的特征向量。

---

## 2. Planning Head 代码精读

**代码位置**：`VAD_head.py:754-839`

### 2.0 先看配置开关（VAD_tiny 实际取值）

不同配置走不同分支，VAD_tiny 的取值（`VAD_tiny_e2e.py:76-112`）决定了实际路径：

```python
ego_his_encoder = None      # → ego query 用可学习的 self.ego_query，不编码历史轨迹
ego_lcf_feat_idx = None     # → ego_feats 不拼 CAN 特征
query_thresh = 0.0          # → 筛选阈值为0，几乎所有agent/map都通过
query_use_fix_pad = False   # → 不强制padding
ego_fut_mode = 3            # → 输出3条候选轨迹（左/直/右）
fut_ts = 6                  # → 每条6步（3秒）
ego_agent_decoder: 1层 cross_attn (CustomTransformerDecoder)
ego_map_decoder:   1层 cross_attn (CustomTransformerDecoder)
```

**意味着 tiny 走最精简路径**：ego query 是纯可学习向量，不吃历史轨迹也不拼 CAN。下面按实际走的分支讲。

### 2.1 第0步：初始化 ego query（756-759行）

```python
if self.ego_his_encoder is not None:
    ego_his_feats = self.ego_his_encoder(ego_his_trajs)  # 编码历史轨迹
else:
    ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)  # ← tiny走这里
```

tiny 配置下，ego query 是**一个可学习的 256 维向量**（`nn.Embedding(1, 256)`），复制到每个 batch。它是一张"白纸"，靠后面的交互填充内容。

> 对比：base 配置用 `ego_his_encoder` 把过去轨迹编码进去，ego query 一开始就带"我刚才怎么开的"信息。tiny 省了这步。

### 2.2 第1步：ego ↔ agent 交互（760-780行）

```python
ego_query = ego_his_feats                          # (B, 1, 256)
ego_pos = torch.zeros((batch, 1, 2))               # ego 永远在原点(0,0)
ego_pos_emb = self.ego_agent_pos_mlp(ego_pos)

# 准备 agent 食材
agent_query = self.agent_fus_mlp(motion_hs.reshape(batch, num_agent, -1))  # (B,300,256)
agent_pos = outputs_coords_bev[-1]                 # (B,300,2)
agent_query, agent_pos, agent_mask = self.select_and_pad_query(...)  # 筛选

# cross-attention：ego 看所有 agent
ego_agent_query = self.ego_agent_decoder(
    query=ego_query,      # ego 作为 query（1个）
    key=agent_query,      # agent 作为 key/value（~300个）
    value=agent_query,
    query_pos=ego_pos_emb,
    key_pos=agent_pos_emb,
    key_padding_mask=agent_mask)
```

**含义**：ego query（1个）去"看"所有 agent（~300个）。cross-attention 让 ego 自动学会**关注哪些车重要**：
- 前方近处要并道的车 → 高注意力
- 后方远处的车 → 低注意力

ego 永远在 `(0,0)`——因为所有坐标都是**自车中心坐标系**，自车就是原点。agent_pos 是相对 ego 的位置，天然就是"那辆车在我哪个方位"。

### 2.3 第2步：ego ↔ map 交互（782-807行）

```python
map_query = self.lane_encoder(map_hs[-1].view(B,100,20,-1))  # (B,100,256)
# ... 算每条线的最近点作为位置 min_map_pos ...
map_query, map_pos, map_mask = self.select_and_pad_query(...)

ego_map_query = self.ego_map_decoder(
    query=ego_agent_query,   # ← 注意！接的是上一步的输出，不是原始ego
    key=map_query,           # 车道线作为 key/value
    value=map_query,
    query_pos=ego_pos_emb,
    key_pos=map_pos_emb,
    key_padding_mask=map_mask)
```

**关键点**：`query=ego_agent_query`——这是**级联**，不是并联。ego 先看完车，带着"周围车的信息"再去看路。

含义：ego 学会关注**相关的车道线**：
- 我当前所在车道 → 高注意力
- 要拐进去的车道 → 高注意力
- 对向无关车道 → 低注意力

经过两步，ego query 已经"看遍了车和路"。

### 2.4 第3步：拼接特征（809-834行）

tiny 配置（`ego_his_encoder=None`, `ego_lcf_feat_idx=None`）走829行分支：

```python
ego_feats = torch.cat(
    [ego_agent_query.permute(1, 0, 2),   # 看完车的特征
     ego_map_query.permute(1, 0, 2)],    # 看完路的特征
    dim=-1
)  # (B, 1, 2D=512)
```

把"车的认知"和"路的认知"拼成 512 维。

四个分支对照（取决于两个开关）：

| ego_his_encoder | ego_lcf_feat_idx | ego_feats 组成 | 维度 |
|---|---|---|---|
| 有 | 有 | his + map + CAN | 2D+2 |
| 有 | 无 | his + map | 2D |
| 无 | 有 | agent + map + CAN | 2D+2 |
| **无（tiny）** | **无（tiny）** | **agent + map** | **2D=512** |

> base 配置会拼上 `ego_his_feats`（历史）和 `ego_lcf_feat`（CAN速度等），维度更高。tiny 只用交互结果。

### 2.5 第4步：解码出轨迹（836-839行）

```python
outputs_ego_trajs = self.ego_fut_decoder(ego_feats)   # MLP: 512 → 3*6*2=36
outputs_ego_trajs = outputs_ego_trajs.reshape(B, 3, 6, 2)
#                                              B, ego_fut_mode, fut_ts, xy
```

一个 MLP 把 512 维特征解码成 **3 条候选轨迹**，每条 6 步、每步 (x,y)。

**为什么是 3 条？** 对应导航指令：左转 / 直行 / 右转。推理时根据真实导航指令 `ego_fut_cmd` 选其中一条：

```python
ego_fut_cmd_idx = argmax(ego_fut_cmd)         # 0/1/2
ego_fut_pred = outputs_ego_trajs[ego_fut_cmd_idx]  # (6, 2)
```

### 2.6 完整链路图

```
ego_query (1个白纸向量, 在原点)
    │
    │  ┌─ key/value: agent_query (300辆车的"位置+意图")
    ↓  ↓
[ego_agent_decoder]  ← ego看车："谁要挡我路？"
    │  cross-attn
    ↓
ego_agent_query (1个, 已含周围车信息)
    │
    │  ┌─ key/value: map_query (100条车道线)
    ↓  ↓
[ego_map_decoder]    ← ego看路："我该沿哪条道走？"
    │  cross-attn
    ↓
ego_map_query (1个, 已含车+路信息)
    │
    ├──── cat(ego_agent_query, ego_map_query) → (B,1,512)
    ↓
[ego_fut_decoder]  MLP
    ↓
3条候选轨迹 (B, 3, 6, 2)
    ↓ 按导航指令选1条
最终轨迹 (6, 2) → 控制器
```

### 2.7 三个设计要点（面试高频）

1. **级联不是并联**：ego 先看车（`ego_agent_decoder`）→ 再看路（`ego_map_decoder` 的 query 接前者输出）。顺序有意义：先知道障碍，再选路径。

2. **ego 恒在原点**：自车中心坐标系下 ego=`(0,0)`，所有 agent/map 位置都是相对量，cross-attention 的位置编码天然表达"相对方位"。

3. **多模态对应导航指令**：3 条轨迹 = 左/直/右，不是随机多样性，是和高层指令绑定的。训练时只监督指令对应的那条（`ego_fut_cmd==1` 筛选）。

---

## 3. select_and_pad_query 筛选机制

**代码位置**：`VAD_head.py:1881-1942`

### 3.0 为什么需要这个函数

第2章里 ego 看车、看路前都先调用它：

```python
agent_query, agent_pos, agent_mask = self.select_and_pad_query(agent_query, agent_pos, agent_conf, ...)
map_query, map_pos, map_mask = self.select_and_pad_query(map_query, min_map_pos, map_conf, ...)
```

解决两个工程问题：

1. **筛选**：300 个检测槽位里大部分是"背景"（没真车），不该让 ego 去关注它们
2. **对齐**：不同 batch 筛出来的有效数量不同（这张图12辆车，那张图30辆），但张量必须形状一致才能 stack

### 3.1 第1段：按置信度筛选（1902-1910行）

```python
query_score = query_score.sigmoid()
query_score = query_score.max(dim=-1)[0]   # 10类里取最高分（"最像某类的程度"）
query_idx = query_score > score_thresh     # 布尔掩码：哪些通过阈值

# 找出整个batch里"有效数量最多"的那张图
batch_max_qnum = 0
for i in range(query_score.shape[0]):
    qnum = query_idx[i].sum()
    if qnum > batch_max_qnum:
        batch_max_qnum = qnum
```

- `max(dim=-1)`：一个 query 输出 10 类分数，取最高的作为"它到底像不像个物体"。背景槽位 10 类都低分 → 被 `score_thresh` 滤掉。
- **注意 tiny 配置 `query_thresh=0.0`**——阈值为0意味着 `sigmoid>0` 永远成立，**几乎所有 300 个都通过**。所以 tiny 实际没真筛，主要靠 padding 对齐。base 配置才会设较高阈值真正筛。
- `batch_max_qnum`：找出 batch 里"最多有效数"，作为统一对齐的长度。

### 3.2 第2段：逐样本筛选 + padding 补齐（1912-1930行）

```python
for i in range(batch):
    valid_query = query[i, query_idx[i]]        # 只留通过的
    valid_query_pos = query_pos[i, query_idx[i]]
    pad_qnum = batch_max_qnum - valid_qnum      # 差多少补多少

    padding_mask = [False] * batch_max_qnum     # 默认全有效
    if pad_qnum != 0:
        valid_query = cat([valid_query, zeros(pad_qnum, dim)])  # 补0
        valid_query_pos = cat([valid_query_pos, zeros(pad_qnum, 2)])
        padding_mask[valid_qnum:] = True         # 补的部分标记为"无效"
```

**核心是 padding_mask**：
```
假设 batch_max_qnum=30
图A有30辆: [v,v,v,...,v]           mask=[F,F,...,F]  全有效
图B有12辆: [v,...,v, 0,0,...,0]    mask=[F,..F, T,T,..T]  后18个是padding
                     ↑补18个0          ↑标记为True
```

`padding_mask` 后续传给 attention 的 `key_padding_mask`——**attention 会跳过 mask=True 的位置**，所以补的 0 不会污染结果。这就是为什么能补 0 还不出错。

### 3.3 第3段：固定填充位兜底（1932-1940行）

```python
if use_fix_pad:   # tiny配置 query_use_fix_pad=False，跳过
    pad_query = torch.zeros((num_batch, 1, feat_dim))
    pad_query_pos = torch.ones((num_batch, 1, 2))
    pad_mask = [False]  # 注意是False，这个pad是"有效"的
    selected_query = cat([selected_query, pad_query], dim=1)   # 额外加1个
```

**兜底机制**：万一某张图一个有效 query 都没有（全是背景），attention 的 key 全空会报错/NaN。所以强制加一个"假 query"（特征全0、位置(1,1)、mask=False表示有效），保证 key 至少有1个。

tiny 配置 `query_use_fix_pad=False` 不启用；base 启用以增强鲁棒性。

### 3.4 返回值

```python
return selected_query, selected_query_pos, selected_padding_mask
#      (B, max_n, D)    (B, max_n, 2)       (B, max_n)
```

正好对应 ego 交互时的三个参数：`key` / `key_pos` / `key_padding_mask`。

### 3.5 流程图

```
输入: 300个agent (含大量背景)
       │
   ① sigmoid + max(10类) → 每个的"物体性"分数
       │
   ② > score_thresh 筛选
       │  图A: 30个有效   图B: 12个有效  （数量不齐！）
       │
   ③ 找 batch_max = 30，所有图补齐到30
       │  图A: 30真         mask=[F×30]
       │  图B: 12真+18假0   mask=[F×12, T×18]
       │
   ④ (可选)再加1个fix_pad兜底
       │
   输出: (B, 30, 256) + mask(B, 30)
       │
   喂给 ego_agent_decoder 的 key/value
       attention 用 mask 跳过假的padding
```

### 3.6 要点（面试）

1. **mask 是关键**：padding 补 0 不会出错，因为 attention 用 `key_padding_mask` 把它们排除在 softmax 之外。这是处理变长序列的标准技巧。

2. **tiny 几乎不筛**：`query_thresh=0.0` 让所有 query 通过，功能退化成"padding对齐"。真正的置信度筛选在 base 配置或推理后处理里。

3. **map 版本多一层距离过滤**：还有个 `select_and_pad_pred_map`（用在 Motion Head），除置信度筛选外，还按距离过滤（`dis_thresh`）——只保留每个 agent 附近的车道线。原理相同，多一层空间裁剪。

---

## 4. 三个规划约束损失

**代码位置**：`utils/plan_loss.py`

这是 VAD 规划的灵魂，也是规则规划背景工程师最容易对接的部分——本质是把驾驶规则写成可微的损失。

### 4.0 共同套路：hinge loss（铰链损失）

三个损失都用同一模式——**"安全不罚，危险才罚，越危险罚越重"**：

```python
safe_idx = loss > thresh      # 距离够远 = 安全
unsafe_idx = loss <= thresh   # 距离太近 = 危险
loss[safe_idx] = 0                        # 安全区：零惩罚
loss[unsafe_idx] = thresh - loss[unsafe]  # 危险区：越近惩罚越大
```

```
惩罚
  │
  │\
  │ \
  │  \
  │   \____________
  └────┴───────────► 距离
      thresh
   危险区  安全区
```

> 这是规则规划里"安全走廊/软约束/势场法"的可微版本。

**共同前置**：`pred.cumsum(dim=-2)`。模型输出每步**位移**(offset)，cumsum 累加成**绝对位置**轨迹才能算距离。

### 4.1 loss_plan_col（碰撞约束）

**目标**：别撞车。`plan_col_loss`，`plan_loss.py:261-316`

**筛选**（forward，239-251行）：
```python
not_valid_agent_mask = agent_max_score < self.agent_thresh  # 滤掉低置信度假目标
agent_fut_preds[not_valid_agent_mask] = 1e6                  # 推到无穷远=忽略
not_veh_pred_mask = agent_max_score_idxs > 4                 # 只防车(类0-4)，不防锥桶
agent_fut_preds[not_veh_pred_mask] = 1e6
best_mode_idxs = argmax(agent_fut_cls_preds)                 # 6模态只取最可能的1条
```

三层过滤：**只防真实的、是车的、最可能走的那条轨迹**。无关目标设 `1e6`，距离巨大，hinge 自然为0。

**核心**：
```python
pred = pred.cumsum(-2)                              # ego绝对轨迹
target = agent_pos + agent_fut_preds.cumsum(-2)     # 每辆车每步绝对位置
dist = norm(ego - target); target[dist>3.0] = 1e6   # 先滤掉3m外远车

x_dist = |ego.x - agent.x|   # hinge thresh=1.5m
y_dist = |ego.y - agent.y|   # hinge thresh=3.0m
```

**关键设计：x/y 阈值不同（1.5m vs 3.0m）**。因为车是长条形：
```
        y (前后，纵向)
        ↑  需要 3.0m 安全距离（刹车距离，追尾风险）
   ─────┼─────→ x (左右，横向)
        │  需要 1.5m 安全距离（车宽，剐蹭风险）
```
纵向留更大余量，横向较小——贴合实际驾驶。

### 4.2 loss_plan_bound（边界约束）

**目标**：别开出路面。`plan_map_bound_loss`，`plan_loss.py:88-144`

**筛选**（forward，72-80行）：
```python
not_lane_bound_mask = lane_score[..., 2] < map_thresh  # 只要"boundary"类(idx=2)
lane_bound_preds[not_lane_bound_mask] = 1e6            # 其他线忽略
```
只关心**道路边界线**（divider分道线、crossing人行道不算边界）。

**核心**：
```python
pred = pred.cumsum(-2)
dist = norm(ego_pos - all_boundary_pts)  # 找最近边界点
min_dist = dist.min()
loss[min_dist > 1.0] = 0          # hinge dis_thresh=1.0m
loss[min_dist <= 1.0] = 1.0 - min_dist
```

**亮点：穿越检测**（122-142行）：
```python
intersect_mask = segments_intersect(ego轨迹段, 边界线段)  # 几何判断线段相交
for 穿越点:
    loss[穿越时刻之后:] = 0   # 穿越点之后清零
```

`segments_intersect`（147-176行）用**叉积+参数方程**判断两线段是否真相交。

为什么穿越后清零？一旦穿过边界，后面的点都在界外，再罚"离边界距离"没意义（越界越远 hinge 反而不罚，逻辑矛盾）。所以从穿越时刻起停止惩罚——**惩罚"接近边界"，不是"已出界后的位置"**。

### 4.3 loss_plan_dir（方向约束）

**目标**：车头朝向和车道方向一致（别斜着开）。`plan_map_dir_loss`，`plan_loss.py:390-447`

**筛选**（forward，374行）：
```python
not_lane_div_mask = lane_score[..., 0] < map_thresh  # 只要"divider"分道线(idx=0)
```
用**分道线**判断行驶方向（边界是路沿，分道线才指示车道走向）。

**核心**：
```python
pred = pred.cumsum(-2)
traj_yaw = atan2(diff(ego.y), diff(ego.x))       # ego轨迹朝向角
lane_yaw = atan2(diff(lane_pts.y), diff(lane_pts.x))  # 最近车道线朝向角
yaw_diff = traj_yaw - lane_yaw
loss = |yaw_diff|

yaw_diff[dist > 2.0] = 0      # 附近没车道线 → 不约束
yaw_diff[static_mask] = 0     # ego几乎没动(<1m) → 朝向无意义，不约束
```

**方向歧义处理**（438-441行）：
```python
yaw_diff[yaw_diff > π/2] -= π   # 车道线点序可能反向标注
yaw_diff[yaw_diff < -π/2] += π
```
车道线点序可能正向或反向（A→B 或 B→A），lane_yaw 差 180°。把超过 ±90° 的差异折叠回来——**只关心"是否平行于车道"，不关心同向反向**。

`static_mask` 豁免：车没动时相邻轨迹点重合，atan2 朝向是噪声，不该约束。

### 4.4 三损失对比表

| | loss_plan_col | loss_plan_bound | loss_plan_dir |
|---|---|---|---|
| **防什么** | 撞车 | 出路面 | 斜着开 |
| **参照物** | 他车(类0-4) | 边界线(idx2) | 分道线(idx0) |
| **阈值** | x:1.5 y:3.0m | 1.0m | π/2(角度) |
| **特殊处理** | x/y分开 | 穿越后清零 | 方向折叠+静止豁免 |
| **来源数据** | Motion预测 | Map预测 | Map预测 |
| **默认权重** | 0.1 | 0.1 | 0.1 |

### 4.5 要点（面试高频）

1. **可微的规则约束**：把"碰撞检测、车道保持、航向对齐"这些规则常识，写成**可反向传播的 hinge loss**，让网络在数据驱动里也守规矩。规则规划背景在此直接对接。

2. **用预测而非GT**（回顾1.7）：`lane_preds`、`agent_fut_preds` 都是 Head 的**预测值**。训练时让 ego 学会"在带误差感知下规避"，保证训练推理一致。

3. **软约束的局限**：是 loss 软约束，非硬保证，不能 100% 防碰撞——这正是 VADv2 引入"轨迹词表+后处理筛选"的动机之一（可加硬规则过滤）。

---

## 5. VADv2 概率规划头

**代码位置**：`VADv2/VADv2_head.py`、`VADv2/VADv2_config_voca4096.py`

> 注意：本仓库的 VADv2 是 **CARLA 适配版**（含红绿灯分类、停止标志、CARLA 导航指令、`carla_plan_vocabulary_4096.npy`），不是原版 nuScenes VADv2。但核心思想一致。

### 5.0 v1 → v2 的本质改变：回归变分类

VADv2 论文的唯一核心贡献，一句话：

> **v1**：直接**回归**出一条轨迹（"我算出来该这么走"）
> **v2**：在 N 条**预定义轨迹**上做**分类**，输出概率分布（"这 4096 条里，每条多大概率是对的"）

```
v1 规划头：
  ego_feats (512) → MLP → 3条轨迹 (3, 6, 2)   ← 直接吐坐标
                          按导航指令选1条

v2 规划头：
  轨迹词表 = 4096条预定义轨迹（固定，离线聚类得到）
  ego_feats → 给每条轨迹打分 → softmax → 概率分布(4096)
                                          argmax选最高分那条
```

### 5.1 轨迹词表（VADv2_head.py:204-205）

```python
self.plan_anchors = np.load(plan_anchors_path)  # 'carla_plan_vocabulary_4096.npy'
self.plan_anchors = torch.from_numpy(...).cuda()  # (4096, 6, 2)
```

4096 条轨迹**提前算好存在文件里**，每条 6 步 ×(x,y)。来源：对训练集所有真实轨迹做**聚类**（类似 k-means），得到 4096 个有代表性的"轨迹原型"，涵盖直行、各种曲率转弯、加减速等常见开法。

配置（`VADv2_config_voca4096.py:97-104`）：
```python
plan_fut_mode=256,           # 训练时采样256条
plan_fut_mode_testing=4096,  # 测试时用全部4096条
plan_anchors_path='carla_plan_vocabulary_4096.npy',
```

### 5.2 轨迹不是预测的，是"贴"上去的（VADv2_head.py:1112-1119）

```python
outputs_ego_trajs = self.plan_reg_branch(ego_feats)         # 先走个回归（占位）
outputs_ego_trajs = outputs_ego_trajs.reshape(B, mode, 6, 2)
outputs_ego_trajs = outputs_ego_trajs * 0. + self.used_plan_anchors[None]  # ← 直接替换成词表
```

看 `* 0. +`——回归输出被**乘0清零**，直接用词表里的轨迹。说明 v2 模型**不生成坐标**，坐标来自固定词表。模型只负责"选哪条"。

### 5.3 多个打分分支（VADv2_head.py:594-616, 1126-1129）

```python
outputs_ego_cls_col    = self.plan_cls_col_branch(ego_feats)     # 碰撞分
outputs_ego_cls_bd     = self.plan_cls_bd_branch(ego_feats)      # 边界分
outputs_ego_cls_cl     = self.plan_cls_cl_branch(ego_feats)      # 中心线分
outputs_ego_cls_expert = self.plan_cls_expert_branch(ego_feats)  # 专家模仿分
```

**v2 把 v1 的三个约束损失升级成了打分维度**。v1 用 loss 软约束一条回归轨迹；v2 给每条候选轨迹打多个分：

- `col`：会不会撞车（对应 v1 的 loss_plan_col）
- `bd`：会不会出边界（对应 loss_plan_bound）
- `cl`：偏不偏离中心线
- `expert`：像不像专家（人类）开的——模仿学习的核心分

**这是 v2 的精髓**：约束从"训练时的 loss"变成了"推理时可见的分数"，可解释、可后处理。

### 5.4 导航指令变成特征（VADv2_head.py:1092-1104）

```python
# [VOID,LEFT,RIGHT,STRAIGHT,LANEFOLLOW,CHANGELANELEFT,CHANGELANERIGHT]
cmdid_onehot = one_hot(command_id)        # 7种CARLA导航指令
ego_cmdid_feat = self.ego_feat_projs[5](cmdid_onehot)
```

v1 的导航指令是"选 3 条里的哪条"；v2 把指令编码成特征喂进 ego_feats，更精细（7 种指令 vs 3 种）。

### 5.5 v1 vs v2 全面对比

| 维度 | VAD v1 | VAD v2 |
|---|---|---|
| **规划范式** | 回归 | 在词表上分类 |
| **输出** | 3条轨迹(左/直/右) | 4096条概率分布 |
| **轨迹来源** | 模型生成坐标 | 固定词表（聚类得到） |
| **约束** | 训练时3个loss软约束 | 推理时多维打分(col/bd/cl/expert) |
| **多模态** | 弱（3条，绑定指令） | 强（4096条，覆盖各种开法） |
| **可解释性** | 弱（黑盒回归） | 中（能看每条概率/各维度分数） |
| **可后处理** | 难 | 易（可加硬规则筛掉危险轨迹） |

### 5.6 为什么是关键改进（面试话术）

> "v1 的回归是单模态的——网络只吐一条轨迹，遇到'左转右转都行'的歧义场景容易学成两者平均（开到中间撞上）。v2 改成在 4096 条预定义轨迹上分类，输出概率分布，天然支持多模态：左转一个峰、右转一个峰，不会平均。
>
> 更重要的是**可控性**。v1 的安全约束是训练时的 loss，推理时看不见、改不了。v2 把碰撞、边界这些约束变成每条候选轨迹的**打分**，推理时可见——可以在选轨迹前用规则硬筛掉所有'碰撞分高'的，再从安全的里选概率最高的。这正好把端到端的学习能力和规则规划的安全兜底结合起来。"

### 5.7 诚实的本质（加分项）

v2 论文里这叫 **Probabilistic Planning**——把规划建模成概率分布 `P(轨迹|场景)`，用专家轨迹做监督（哪条词表轨迹最接近真实轨迹，那条标签为正）。本质是**模仿学习 + 离散动作空间**。

> 这和强化学习里的离散动作空间选择很像——也是为什么后续 RAD 能在这个框架上加强化学习后训练。轨迹词表给 RL 提供了天然的离散动作集。

---

## 6. 小结：规划模块全景

```
                  感知/预测产物
    agent(意图+位置+conf)    map(车道+位置+conf)    ego(历史+CAN)
              │                    │                  │
              └──────────┬─────────┴──────────────────┘
                         ↓
              ┌──────────────────────┐
              │  ego query 级联交互   │
              │  ego↔agent → ego↔map  │
              └──────────┬───────────┘
                         ↓
          ┌──────────────┴───────────────┐
          ↓                               ↓
    【v1】回归头                    【v2】分类头
    ego_fut_decoder                 在4096词表上打分
    → 3条轨迹(左/直/右)             → 概率分布 + 多维评分
          │                               │
    训练: 3个约束loss               推理: 可后处理筛选
    (col/bound/dir 软约束)          (col/bd/cl/expert 评分)
          │                               │
          └───────────┬───────────────────┘
                      ↓
                 最终轨迹 (6,2) → 控制器
```

**三句话记住整章**：
1. 规划吃的是**预测值**（agent/map 都是 Head 预测，非 GT），位置都 detach。
2. v1 是 **ego query 级联交互（先看车再看路）+ 回归轨迹 + 3个约束 loss**。
3. v2 是 **轨迹词表分类 + 多维打分**，多模态更强、可解释、可加规则后处理。




