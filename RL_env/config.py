"""
本文件作为全局静态配置。
将节点规模、硬件资源界限、网络物理属性统一定义。
测试系统在不同算力密度下的鲁棒性时，只需修改此处即可，无需侵入底层环境逻辑。
"""

# 物理拓扑节点配置
NUM_CLOUD_NODES = 1
NUM_EDGE_NODES = 10
NUM_NODES = NUM_CLOUD_NODES + NUM_EDGE_NODES

# 边缘节点资源容量范围
EDGE_CPU_RANGE = (16, 64)  # CPU核心数范围
EDGE_MEM_RANGE = (16, 128)  # 内存容量范围（GB）
EDGE_GPU_RANGE = (0, 2)  # 部分边缘节点可能配备GPU，范围为0-2

# 云节点(拥有海量资源，索引设为 0)
CLOUD_CPU = 1024.0  # 云节点CPU核心数
CLOUD_MEM = 2048.0  # 云节点内存容量（GB）
CLOUD_GPU = 128.0  # 云节点GPU数量

# 物理链路通信配置
BANDWIDTH_RANGE = (50.0, 100.0)  # 链路带宽范围（Mbps）
LATENCY_RANGE = (0.2, 2.0)  # 链路延迟范围（ms）