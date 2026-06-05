# nuScenes HD Map 数据格式详解

> 解答：HD Map是什么格式？如何表示车道线、边界、人行道？

---

## 🎯 核心答案

**是的！nuScenes的HD Map就是"点坐标+属性"的集合。**

具体来说：
- ✅ **点坐标**：每个地图元素是一串有序的(x, y, z)点
- ✅ **属性**：类型（车道线/边界/人行道）、token、连接关系

---

## 📊 nuScenes HD Map 的完整结构

### 1. 地图数据的存储格式

```
nuScenes 地图文件：
data/nuscenes/maps/
├── basemap/
│   ├── boston-seaport.png          # 底图（可视化用）
│   ├── singapore-onenorth.png
│   └── ...
└── expansion/
    ├── boston-seaport.json         # 矢量地图（VAD使用这个）
    ├── singapore-onenorth.json
    └── ...
```

**VAD使用的是 `expansion/*.json` 矢量地图文件！**

---

### 2. JSON文件的结构

```json
{
  "lane": [
    {
      "token": "lane_token_123",           // 唯一ID
      "polygon_token": "polygon_456",      // 对应的多边形区域
      "from_edge_line_token": "edge_789",  // 起始边线
      "to_edge_line_token": "edge_101",    // 结束边线
      ...
    },
    ...
  ],
  
  "lane_divider": [                        // 车道分隔线
    {
      "token": "divider_token_001",
      "line_token": "line_002",            // 指向实际的线条
      ...
    },
    ...
  ],
  
  "road_segment": [...],                   // 道路片段
  "road_block": [...],                     // 道路块
  "ped_crossing": [...],                   // 人行横道
  "walkway": [...],                        // 人行道
  "stop_line": [...],                      // 停止线
  "carpark_area": [...],                   // 停车区
  
  "line": [                                // ⭐ 实际的点坐标在这里
    {
      "token": "line_002",
      "line_type": "SOLID",                // 实线/虚线
      "node_tokens": [                     // 指向节点（点坐标）
        "node_a1",
        "node_a2",
        "node_a3",
        ...
      ]
    },
    ...
  ],
  
  "node": [                                // ⭐⭐ 最底层：点坐标
    {
      "token": "node_a1",
      "x": 1025.3,                         // global坐标系的x（米）
      "y": 543.7,                          // global坐标系的y（米）
      "z": 0.0                             // 高度（通常为0）
    },
    {
      "token": "node_a2",
      "x": 1027.5,
      "y": 544.1,
      "z": 0.0
    },
    ...
  ],
  
  "polygon": [                             // 区域多边形
    {
      "token": "polygon_456",
      "exterior_node_tokens": [            // 外围点
        "node_b1", "node_b2", ...
      ],
      "holes": []                          // 内部空洞
    },
    ...
  ]
}
```

---

## 🔍 地图元素的层级结构

```
高层语义
    ↓
lane_divider（车道分隔线）
    ↓ line_token
line（线条，包含类型）
    ↓ node_tokens
node（实际点坐标）

示例：
lane_divider_001
    → line_token: "line_abc"
        → line: {type: "SOLID", node_tokens: ["node1", "node2", "node3"]}
            → node1: {x: 100.0, y: 200.0, z: 0.0}
            → node2: {x: 101.5, y: 201.2, z: 0.0}
            → node3: {x: 103.0, y: 202.5, z: 0.0}
```

---

## 🎨 VAD使用的3类地图元素

### 类别1：lane_divider（车道分隔线）

**物理含义**：分隔两条车道的线（实线或虚线）

**数据结构**：
```json
{
  "token": "lane_divider_123",
  "line_token": "line_456",           // 指向实际线条
  "lane_tokens": [                    // 这条线分隔哪两条lane
    "lane_left", 
    "lane_right"
  ]
}
```

**对应的line**：
```json
{
  "token": "line_456",
  "line_type": "SOLID",               // 实线（不可变道）或 "DASH"（虚线，可变道）
  "node_tokens": [
    "node_a", "node_b", "node_c", ... // 一串有序的点
  ]
}
```

**最终得到**：一条polyline，由10-50个点组成

---

### 类别2：road_boundary（道路边界）

**物理含义**：道路的边缘（路肩、护栏、路缘石）

**数据结构**：
```json
{
  "token": "boundary_789",
  "line_token": "line_def",
  "road_segment_token": "segment_001" // 属于哪个道路片段
}
```

**特点**：
- 通常是实线类型
- 标记车辆不能越过的边界
- VAD用来计算"越界约束"

---

### 类别3：ped_crossing（人行横道）

**物理含义**：斑马线

**数据结构**：
```json
{
  "token": "crossing_abc",
  "polygon_token": "poly_xyz",        // 人行横道是一个区域（多边形）
  "road_segment_token": "segment_001"
}
```

**对应的polygon**：
```json
{
  "token": "poly_xyz",
  "exterior_node_tokens": [
    "node_1", "node_2", "node_3", "node_4" // 4个角点
  ],
  "holes": []
}
```

**最终得到**：一个矩形区域，用4个角点表示

---

## 🛠️ VAD如何处理地图数据

### 步骤1：查询当前区域的地图元素

```python
# 伪代码：预处理脚本中的地图查询
from nuscenes.map_expansion.map_api import NuScenesMap

# 初始化地图
nusc_map = NuScenesMap(dataroot='data/nuscenes', map_name='boston-seaport')

# 查询ego周围60m×30m范围内的地图元素
ego_x, ego_y = 1025.0, 545.0
patch_box = (ego_x - 30, ego_y - 15, ego_x + 30, ego_y + 15)  # [x_min, y_min, x_max, y_max]

# 获取车道分隔线
lane_dividers = nusc_map.get_records_in_patch(
    patch_box, 
    layer_names=['lane_divider'],
    mode='intersect'  # 与区域相交的元素
)

# 获取道路边界
boundaries = nusc_map.get_records_in_patch(
    patch_box,
    layer_names=['road_segment'],  # 道路边界包含在road_segment中
    mode='intersect'
)

# 获取人行横道
crossings = nusc_map.get_records_in_patch(
    patch_box,
    layer_names=['ped_crossing'],
    mode='intersect'
)
```

---

### 步骤2：提取点坐标

```python
# 对于lane_divider（线条）
def get_lane_divider_points(nusc_map, divider_token):
    """提取车道分隔线的点坐标"""
    # 1. 获取divider记录
    divider_record = nusc_map.get('lane_divider', divider_token)
    
    # 2. 获取对应的line
    line_token = divider_record['line_token']
    line_record = nusc_map.get('line', line_token)
    
    # 3. 获取所有node的坐标
    node_tokens = line_record['node_tokens']
    points = []
    for node_token in node_tokens:
        node = nusc_map.get('node', node_token)
        points.append([node['x'], node['y']])  # 提取(x, y)坐标
    
    return np.array(points)  # shape: [N_points, 2]

# 示例输出
# array([[1025.3, 543.7],
#        [1027.5, 544.1],
#        [1029.8, 544.6],
#        ...
#        [1085.2, 567.3]])  # 一条线有10-50个点
```

---

### 步骤3：采样为固定数量的点

**问题**：不同的地图元素点数不同（10-50个），模型需要固定长度

**解决**：采样/插值为20个点

```python
def resample_polyline(points, num_samples=20):
    """将polyline重采样为固定数量的点"""
    if len(points) < 2:
        # 点太少，填充
        return np.pad(points, ((0, num_samples - len(points)), (0, 0)))
    
    # 计算累积弧长
    diffs = np.diff(points, axis=0)
    segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_lengths[-1]
    
    # 均匀采样20个点
    sample_positions = np.linspace(0, total_length, num_samples)
    
    # 线性插值
    resampled_points = np.zeros((num_samples, 2))
    for i, pos in enumerate(sample_positions):
        # 找到pos对应的线段
        idx = np.searchsorted(cumulative_lengths, pos) - 1
        idx = max(0, min(idx, len(points) - 2))
        
        # 线段内插值
        t = (pos - cumulative_lengths[idx]) / segment_lengths[idx]
        resampled_points[i] = points[idx] + t * (points[idx + 1] - points[idx])
    
    return resampled_points  # shape: [20, 2]
```

---

### 步骤4：坐标变换

**从global坐标系转到ego坐标系**

```python
def transform_to_ego(points, ego_pose):
    """global坐标 → ego坐标"""
    ego_x, ego_y, ego_yaw = ego_pose
    
    # 1. 平移：减去ego位置
    points = points - np.array([ego_x, ego_y])
    
    # 2. 旋转：对齐到ego朝向
    cos_yaw = np.cos(-ego_yaw)
    sin_yaw = np.sin(-ego_yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw,  cos_yaw]
    ])
    points = points @ rotation_matrix.T
    
    return points

# 示例：
# global坐标：[[1025.3, 543.7], [1027.5, 544.1], ...]
# ego在：(1020.0, 540.0, yaw=0.1)
# 转换后：[[5.3, 3.7], [7.5, 4.1], ...]（相对ego的位置）
```

---

### 步骤5：保存到.pkl

```python
# 预处理脚本最终生成的格式
info['map_anns'] = {
    'lane_divider': [
        np.array([[0.5, 1.2], [1.3, 1.5], ..., [19.8, 5.3]]),  # [20, 2]
        np.array([[0.2, -2.1], [1.1, -2.0], ..., [20.1, -1.5]]),
        ...  # 多条车道线
    ],
    'road_boundary': [
        np.array([[-15.0, 10.0], [-14.5, 10.1], ..., [15.0, 10.0]]),  # [20, 2]
        np.array([[-15.0, -10.0], [-14.5, -10.0], ..., [15.0, -10.0]]),
        ...  # 左右边界
    ],
    'ped_crossing': [
        np.array([[10.0, -2.0], [12.0, -2.0], [12.0, 2.0], [10.0, 2.0]]),  # [4, 2] 矩形
        ...  # 人行横道
    ]
}
```

---

## 📊 完整的数据流：HD Map

```
┌──────────────────────────────────────────────────┐
│ nuScenes原始地图：expansion/*.json                │
│                                                   │
│ 结构：                                             │
│   lane_divider → line → node (x,y,z)             │
│   road_boundary → line → node (x,y,z)            │
│   ped_crossing → polygon → node (x,y,z)          │
│                                                   │
│ 特点：                                             │
│   • 点数不固定（10-50个）                         │
│   • global坐标系                                  │
│   • 层级结构（语义 → 几何）                       │
└────────────────┬──────────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────────┐
│ 预处理脚本：vad_nuscenes_converter.py             │
│                                                   │
│ 处理：                                             │
│  1. 查询ego周围60m×30m的地图元素                  │
│  2. 提取点坐标（node的x,y）                       │
│  3. 重采样为20个点（polyline）                    │
│  4. 坐标变换：global → ego                        │
│  5. 保存到.pkl                                    │
└────────────────┬──────────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────────┐
│ .pkl文件：info['map_anns']                        │
│                                                   │
│ 格式：                                             │
│   {                                               │
│     'lane_divider': [                            │
│       np.array([20, 2]),  # 第1条车道线           │
│       np.array([20, 2]),  # 第2条车道线           │
│       ...                                         │
│     ],                                            │
│     'road_boundary': [...],                      │
│     'ped_crossing': [...]                        │
│   }                                               │
│                                                   │
│ 特点：                                             │
│   • 固定20个点（polyline）                        │
│   • ego坐标系                                     │
│   • 只保留几何（点坐标）                          │
└────────────────┬──────────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────────┐
│ VAD训练：使用矢量化地图GT                          │
│                                                   │
│ 模型输入：                                         │
│   • 图像 → BEV特征                                │
│                                                   │
│ 模型输出：                                         │
│   • map_queries → 预测的polyline [N_map, 20, 2]  │
│                                                   │
│ 损失计算：                                         │
│   L_map = chamfer_distance(预测polyline, GT polyline) │
│                                                   │
│ 约束使用：                                         │
│   • ego轨迹不能越过road_boundary                  │
│   • 计算ego到boundary的最短距离                   │
└──────────────────────────────────────────────────┘
```

---

## 💡 关键理解

### 1. **地图是矢量格式，不是栅格**

```
❌ 不是这样：
    0 0 0 1 1 0 0 0  # 栅格图，每个格子表示是否有车道线
    0 0 0 1 1 0 0 0
    
✅ 而是这样：
    lane_divider_1: [(0.5, 1.2), (1.3, 1.5), ..., (19.8, 5.3)]  # 点序列
    lane_divider_2: [(0.2, -2.1), (1.1, -2.0), ..., (20.1, -1.5)]
```

**优势**：
- 精度高（连续坐标，不受栅格分辨率限制）
- 稀疏（只存储有地图元素的地方）
- 语义明确（每条线是独立的实例）

---

### 2. **层级结构：语义 → 几何**

```
语义层（高层）：lane_divider（车道分隔线）
    ↓
几何层（中层）：line（线条，带类型）
    ↓
坐标层（底层）：node（实际点坐标）
```

**为什么分层**：
- 语义层：便于查询（"找所有车道线"）
- 几何层：便于复用（多个lane_divider可以共享同一条line）
- 坐标层：最底层数据

---

### 3. **VAD只用几何，不用语义**

```python
# VAD不关心"这是lane_divider还是road_boundary"
# 只关心"这是一条20个点的polyline"

map_gt_bboxes_3d = [
    polyline_1,  # [20, 2]
    polyline_2,  # [20, 2]
    ...
]

map_gt_labels_3d = [
    0,  # lane_divider
    0,  # lane_divider
    1,  # road_boundary
    2,  # ped_crossing
    ...
]
```

VAD训练时：
- 用polyline计算loss（几何匹配）
- 用label计算分类loss（语义匹配）

---

## 🎓 面试时如何回答

**Q: nuScenes的HD Map是什么格式？**
> "nuScenes的HD Map是矢量格式的JSON文件，采用层级结构。顶层是语义元素（lane_divider、road_boundary等），中间层是几何线条（line），底层是实际的点坐标（node）。每个地图元素是一串有序的(x,y)点，数量不固定。VAD预处理时会查询ego周围的地图元素，提取点坐标，重采样为20个点的polyline，转换到ego坐标系，用作矢量化地图的GT。"

**Q: 为什么用矢量而不是栅格？**
> "矢量格式有三个优势：1）精度高，连续坐标不受分辨率限制；2）稀疏，只存储有地图元素的地方，计算高效；3）语义明确，每条线是独立实例，方便计算约束。比如计算ego到road_boundary的距离，矢量可以直接用点到线的几何公式，精度在厘米级。如果是栅格，要么分辨率低（不准确），要么分辨率高（内存爆炸）。"

---

## 📋 总结

### ✅ 正确理解

- nuScenes HD Map = **点坐标 + 属性（类型）**
- 存储格式：JSON（层级结构）
- 点数：不固定（10-50个）
- 坐标系：global（绝对坐标）

### VAD使用流程

1. 查询ego周围区域的地图元素
2. 提取点坐标（从层级结构中递归查询）
3. 重采样为20个点（统一格式）
4. 坐标变换（global → ego）
5. 用作矢量化地图的GT

### 关键优势

- 矢量格式：稀疏、高精度、实例分离
- 适合计算几何约束
- 适合端到端学习

---

## 🔗 相关文档

- **VAD数据使用说明.md**：VAD使用哪些数据
- **数据处理与闭环系统.md**：完整数据流
- **数据处理与特征提取指南.md**：数据预处理流程

