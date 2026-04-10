"""
文件作用：全局静态配置文件。
包含：节点规模、硬件资源界限、网络物理属性及服务配置的统一定义。
工程规范：本文件只存储常量（Constants），不包含任何业务逻辑函数。
当需要进行消融实验（如动态扩缩容边缘节点规格）时，直接在此处调整阈值即可。
"""

# ================= 物理拓扑节点配置 =================
NUM_CLOUD_NODES = 1
NUM_EDGE_NODES = 10
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
LATENCY_RANGE = (0.2, 2.0)  # 链路延迟范围（ms）

# ================= 服务实体与动作空间配置 =================
NUM_MICROSERVICES = 20  # 微服务的种类数量
NUM_AI_SERVICES = 4  # AI服务种类数量
NUM_SERVICES = NUM_MICROSERVICES + NUM_AI_SERVICES  # 总服务种类数量

# 部署动作空间约束
MAX_INSTANCES = 5  # 每个服务在单一节点的最大实例数
INVALID_ACTION_PENALTY = -100.0  # 智能体选择非法部署动作时的惩罚分数

# ================= 宏观部署动作空间 (Macro-step Deployment) =================
# 采用相对增减指令，取代原先的绝对实例数量。
# 0: 缩容 (-1 实例)
# 1: 保持 (0 实例)
# 2: 扩容 (+1 实例)
DEPLOY_ACTION_DIM = 3  # 每个服务的动作维度（缩容、保持、扩容）

# ================= 动态流量与路由调度配置=================
NUM_FLOWS = 5         # 全网并行的独立请求流(业务调用链)数量
MIN_CHAIN_LENGTH = 3  # 每条请求链最少包含的服务跳数
MAX_CHAIN_LENGTH = 6  # 每条请求链最多包含的服务跳数

ROUTING_ALPHA = 0.5  # 自适应路由权重：1.0 完全偏向目标节点实例容量，0.0 完全偏向最短物理延迟
ROUTING_GAMMA = 0.1  # 物理延迟衰减系数，用于放大或缩小距离对概率分配的影响

MAX_ARRIVAL_RATE = 1500.0  # 单个请求流的初始最大泊松到达率 (req/s)
MAX_DATA_MB = 3.0          # 服务间传递的依赖数据量上限 (MB)