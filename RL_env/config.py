"""
文件作用：全局静态配置文件。
包含：节点规模、硬件资源界限、网络物理属性及服务配置的统一定义。
工程规范：本文件只存储常量（Constants），不包含任何业务逻辑函数。
当需要进行消融实验（如动态扩缩容边缘节点规格）时，直接在此处调整阈值即可。
"""

# ================= 物理拓扑节点配置 =================
NUM_CLOUD_NODES = 1
NUM_EDGE_NODES = 9
NUM_NODES = NUM_CLOUD_NODES + NUM_EDGE_NODES

# ================= 边缘与云节点资源容量范围 =================
EDGE_CPU_RANGE = (16, 64)  # CPU核心数范围
EDGE_MEM_RANGE = (16, 128)  # 内存容量范围（GB）
EDGE_GPU_RANGE = (0, 2)  # 部分边缘节点可能配备GPU，范围为0-2

CLOUD_CPU = 1024.0  # 云节点CPU核心数
CLOUD_MEM = 2048.0  # 云节点内存容量（GB）
CLOUD_GPU = 128.0  # 云节点GPU数量

# ================= 物理链路通信配置 =================
BANDWIDTH_RANGE = (50.0, 100.0)  # 链路带宽范围（Mbps）
EDGE_LATENCY_RANGE = (1.0, 5.0)  # 边缘局域网链路延迟范围（ms)
CLOUD_LATENCY_RANGE = (20.0, 50.0)  # 云端链路延迟范围（ms）

# ================= 服务实体与动作空间配置 =================
NUM_MICROSERVICES = 6  # 微服务的种类数量
NUM_AI_SERVICES = 2  # AI服务种类数量
NUM_SERVICES = NUM_MICROSERVICES + NUM_AI_SERVICES  # 总服务种类数量

# 部署动作空间约束
MAX_INSTANCES = 10  # 每个服务在单一节点的最大实例数
INVALID_ACTION_PENALTY = -100.0  # 智能体选择非法部署动作时的惩罚分数

# ================= 宏观部署动作空间 (Macro-step Deployment) =================
# 采用相对增减指令，取代原先的绝对实例数量。
# 0: 缩容 (-1 实例)
# 1: 保持 (0 实例)
# 2: 扩容 (+1 实例)
DEPLOY_ACTION_DIM = 3  # 每个服务的动作维度（缩容、保持、扩容）

# ================= 动态流量与路由调度配置=================
NUM_FLOWS = 4         # 全网并行的独立请求流(业务调用链)数量
MIN_CHAIN_LENGTH = 3  # 每条请求链最少包含的服务跳数
MAX_CHAIN_LENGTH = 6  # 每条请求链最多包含的服务跳数

ROUTING_ALPHA = 0.5  # 自适应路由权重：1.0 完全偏向目标节点实例容量，0.0 完全偏向最短物理延迟
ROUTING_GAMMA = 0.1  # 物理延迟衰减系数，用于放大或缩小距离对概率分配的影响

MAX_ARRIVAL_RATE = 800.0  # 单个请求流的初始最大泊松到达率 (req/s)
MAX_DATA_MB = 0.1          # 服务间传递的依赖数据量上限 (MB)

# ================= 排队论与延迟惩罚配置 =================
MAX_DELAY_MS = 10000.0  # 超过此延迟视为系统崩溃，给予极大负奖励
MICROSERVICE_MU_RANGE = (100.0, 300.0)  # 边缘节点微服务实例的处理速率范围 (req/s)
AI_SERVICE_MU_RANGE = (50.0, 150.0)      # 边缘节点AI服务实例的处理速率范围 (req/s)

# 云端算力倍乘系数（体现云端海量算力的碾压优势）
CLOUD_MU_MULTIPLIER = 5000.0

# ================= 多目标优化与奖励函数配置 =================
# 优化目标权重 (\eta_1, \eta_2, \eta_3)
ETA_DELAY = 0.5 # 延迟的惩罚权重
ETA_VARIANCE = 0.2 # 负载均衡方差的惩罚权重
ETA_COST = 0.3 # 部署成本的惩罚权重

# 指数衰减特征尺度 (Scale)
# 物理意义：当实际指标达到此值时，该项得分为 exp(-1) ≈ 36%
SCALE_DELAY_MS = 2000.0   
SCALE_VAR = 0.25          
SCALE_COST = 80.0

# SLA (Service Level Agreement) 与 软约束惩罚
TARGET_SLA_SUCCESS = 0.99 # 目标请求成功率 (99%)
QOS_DELAY_TOLERANCE = 500.0 # 应用请求可容忍的最大延迟界限 (ms)，超过即算失败

# 线性惩罚系数
PENALTY_PER_TRUNCATED = 0.2  # 每产生1次非法越界扩容，扣除0.2分
PENALTY_PER_THRASH = 0.1     # 每产生1次AI实例抖动启停，扣除0.1分

# 成本单价定义 (相对单位)
COST_MICROSERVICE = 0.1   # 微服务单实例成本
COST_AI_LIGHT = 2.0       # 轻量级 AI 单实例成本
COST_AI_FULL = 10.0       # 全量级 AI 单实例成本
COST_NODE_ACTIVE = 1.0    # 边缘节点激活(开机且托管了实例)的基础固定成本

# ================= Wrapper 归一化极值基准 =================
# 用于将宏观拓扑的带宽与延迟强行压缩至 [0.0, 1.0] 区间，防止神经网络梯度爆炸
MAX_NORM_BANDWIDTH = 100.0  
MAX_NORM_LATENCY = 100.0 