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