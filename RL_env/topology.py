"""
本文件构建底层物理拓扑类 PhysicalNetwork。
它不处理强化学习的步进逻辑，只负责维护“世界的客观物理状态”，包括节点资源的上限矩阵、
当前可用资源的实时矩阵、以及节点间的带宽与延迟图谱。矩阵化设计是为了后续计算资源消耗时，
可以直接利用向量点积瞬间完成验证，拒绝低效的 for 循环。
"""
import numpy as np
import config

class PhysicalNetwork:
    def __init__(self):
        #记录系统中总的节点数量
        self.num_nodes = config.NUM_NODES

        #初始化核心状态矩阵，这将在 reset() 中被具体赋值
        self.capacity_matrix = None # 节点最大资源容量矩阵 (NUM_NODES, 3)
        self.remain_matrix = None # 节点当前可用资源矩阵 (NUM_NODES, 3)
        self.bandwidth_matrix = None # 节点间带宽矩阵 (NUM_NODES, NUM_NODES)
        self.latency_matrix = None # 节点间延迟矩阵 (NUM_NODES, NUM_NODES)

        #类的实例化时直接进行一次拓扑重置
        self.reset_topology()
    
    def reset_topology(self):
        """
        重置整个物理网络的资源和链路状态，供强化学习每个 Episode 开始时调用。
        """
        # ================= 1. 初始化资源矩阵 =================
        # 创建一个空矩阵，维度为 (节点总数, 3)，3 代表 [CPU, GPU, MEM] 三个维度
        self.capacity_matrix = np.zeros((self.num_nodes, 3), dtype=np.float32)

        # 云节点资源容量设置（索引0）
        self.capacity_matrix[0] = [config.CLOUD_CPU, config.CLOUD_GPU, config.CLOUD_MEM]

        # 边缘节点资源容量随机生成
        for i in range(1, self.num_nodes):
            cpu = np.random.uniform(*config.EDGE_CPU_RANGE)
            #这里*可以传递一个元组作为参数，等价于 cpu = np.random.uniform(config.EDGE_CPU_RANGE[0], config.EDGE_CPU_RANGE[1])
            gpu = np.random.randint(config.EDGE_GPU_RANGE[0], config.EDGE_GPU_RANGE[1] + 1) # GPU数量是整数
            mem = np.random.uniform(*config.EDGE_MEM_RANGE)
            self.capacity_matrix[i] = [cpu, float(gpu), mem]#注意GPU数量也要转换为float类型，以保持矩阵数据类型一致
        
        # 初始时可用资源矩阵与容量矩阵相同，表示所有资源都未被占用
        self.remain_matrix = np.copy(self.capacity_matrix) #np.copy()函数用于创建一个数组的副本
        # 确保capacity_matrix和remain_matrix是两个独立的矩阵,修改remain_matrix不会影响capacity_matrix

        # ================= 2. 初始化网络链路矩阵 =================
        # 同一节点内带宽为无穷大，传播延迟为 0

        #随机生成节点间的带宽矩阵
        self.bandwidth_matrix = np.random.uniform(
            config.BANDWIDTH_RANGE[0], config.BANDWIDTH_RANGE[1], 
            size=(self.num_nodes, self.num_nodes)).astype(np.float32)
        #生成一个随机带宽矩阵，维度为 (NUM_NODES, NUM_NODES)，值在 BANDWIDTH_RANGE 范围内
        #对角线的带宽设置为无穷大，表示同一节点内通信不受带宽限制
        np.fill_diagonal(self.bandwidth_matrix, np.inf) 
        #fill_diagonal函数将矩阵的对角线元素设置为指定值，第一个参数是要修改的矩阵，第二个参数是要设置的值，np.inf表示无穷大

        #随机生成节点间的延迟矩阵
        self.latency_matrix = np.random.uniform(
            config.LATENCY_RANGE[0], config.LATENCY_RANGE[1], 
            size=(self.num_nodes, self.num_nodes)).astype(np.float32)
        #对角线的延迟设置为 0，表示同一节点内通信延迟为 0
        np.fill_diagonal(self.latency_matrix, 0) 