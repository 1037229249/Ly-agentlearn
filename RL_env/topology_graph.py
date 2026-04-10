"""
文件作用：物理网络与拓扑图结构基座。
负责维护“宏观资源拓扑状态”，为图神经网络（GNN）提供标准的节点特征矩阵和边特征张量。
设计逻辑：将离散的带宽与延迟矩阵合并为一个三维张量 E，为后续 GNN 的边属性传递（Edge Attributes）铺平道路。
"""
import numpy as np
import config

class PhysicalNetworkGraph:
    def __init__(self):
        #记录系统中总的节点数量
        self.num_nodes = config.NUM_NODES

        #初始化核心状态矩阵，这将在 reset() 中被具体赋值
        self.capacity_matrix = None # 节点最大资源容量矩阵 (NUM_NODES, 3)
        self.remain_matrix = None # 节点当前可用资源矩阵 (NUM_NODES, 3)
        self.edge_features = None       # 边特征综合张量 (NUM_NODES, NUM_NODES, 2)

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

        # 2. 构造支持 GNN 的三维边特征张量 (NUM_NODES, NUM_NODES, 2)
        # 通道 0 存储带宽 B_{i,j}，通道 1 存储传播延迟 d_{i,j}
        self.edge_features = np.zeros((self.num_nodes, self.num_nodes, 2), dtype=np.float32)

        #填充通道0：带宽矩阵
        #uniform函数用于生成指定范围内的随机数，size参数指定输出数组的形状，
        # 这里生成一个 (NUM_NODES, NUM_NODES) 的矩阵，值在 BANDWIDTH_RANGE 范围内
        self.edge_features[:,:,0] = np.random.uniform(
            config.BANDWIDTH_RANGE[0], config.BANDWIDTH_RANGE[1], 
            size=(self.num_nodes, self.num_nodes)).astype(np.float32)
        
        #同一节点内带宽设置为无穷大，表示通信不受带宽限制
        #fill_diagonal函数将矩阵的对角线元素设置为指定值，第一个参数是要修改的矩阵，第二个参数是要设置的值，np.inf表示无穷大
        np.fill_diagonal(self.edge_features[:,:,0], np.inf)

        #填充通道1：延迟矩阵
        self.edge_features[:,:,1] = np.random.uniform(
            config.LATENCY_RANGE[0], config.LATENCY_RANGE[1], 
            size=(self.num_nodes, self.num_nodes)).astype(np.float32)
        
        #同一节点内延迟设置为0，表示不产生传播延迟
        np.fill_diagonal(self.edge_features[:,:,1], 0.0)

    def get_utilization_matrix(self):
        """
        计算并输出节点维度的资源利用率矩阵 U_t。
        计算逻辑：(总容量 - 剩余容量) / 总容量，输出形状 (NUM_NODES, 3)，值域 [0, 1]。
        """
        comsumed = self.capacity_matrix - self.remain_matrix
        # 防止除零异常,clip函数用于将输入数组中的元素限制在指定的最小值和最大值之间，这里将利用率限制在0.0到1.0之间
        #如果超出范围则会被截断为边界值，确保输出的利用率矩阵中的每个元素都在0.0到1.0之间，避免出现负数或大于1的值。
        utilization = np.clip(comsumed / (self.capacity_matrix + 1e-9), 0.0, 1.0)
        if np.any(utilization < 0.0) or np.any(utilization > 1.0):
            raise ValueError("利用率超过[0, 1]范围，可能存在计算错误，请检查 capacity_matrix 和 remain_matrix 的值。")
        #astype(np.float32)确保输出矩阵的数据类型为 float32，以节省内存并提高计算效率
        return utilization.astype(np.float32)