"""
文件作用：快时间尺度 (Micro-step) 流量生成与路由决策引擎。
包含模块：
1. DynamicTrafficGenerator：基于泊松过程构建有向服务链（DAG）与潮汐流量注入，强加尾端 AI 拓扑约束。
2. HeuristicRouter：基于纯 Numpy 张量运算，毫秒级推导实例感知的自适应转发概率矩阵，并进行流量的聚合链式计算。
"""
import numpy as np
import config

class DynamicTrafficGenerator:
    """
    作用：生成模拟真实网络环境的微服务-AI混合服务调用链与动态泊松到达率。
    严格保证调用链尾端必为 AI 服务，为计算卸载做绝对的物理锚点铺垫。
    """
    def __init__(self):
        self.num_flows = config.NUM_FLOWS
        self.active_flows = []  # 当前活跃的流列表
        # 初始化服务间的依赖数据传输量矩阵 (NUM_SERVICES, NUM_SERVICES)
        # 用于后续计算通信延迟
        self.data_dep_matrix = np.zeros((config.NUM_SERVICES, config.NUM_SERVICES), dtype=np.float32)

        self._generate_service_chains()
        self._init_data_dependencies() 
    
    def _generate_service_chains(self):
        """
        随机生成具有逻辑依赖关系的服务调用链
        """
        self.active_flows = []
        for f in range(self.num_flows):
            # 随机决定该流的链条长度
            chain_length = np.random.randint(config.MIN_CHAIN_LENGTH, config.MAX_CHAIN_LENGTH + 1)

            # 随机挑选服务组装为调用链 (DAG的线性退化形态)
            # 保证业务流一定包含微服务（前排）和 AI 服务（核心）
            # np.random.choice 用于从给定的数组中随机抽取指定数量的元素
            # size 参数指定要抽取的元素数量，replace=True 表示抽取时允许重复，False 则表示不允许重复。
            micro_nodes = np.random.choice(config.NUM_MICROSERVICES, size=chain_length-1, replace=True)
            # 抽取一个 AI 服务作为调用链的绝对尾端 (Sink Node)
            # range表示所有 AI 服务的索引范围。
            ai_node = np.random.choice(range(config.NUM_MICROSERVICES, config.NUM_SERVICES), size=1)
            # 组装为调用链 (如 [网关(微), 鉴权(微), 数据处理(微), 目标检测(AI)])
            chain = np.concatenate((micro_nodes, ai_node))
            # 随机指定流的初始接入边缘节点（排除拥有海量资源的云节点 0）
            start_node = np.random.randint(1, config.NUM_NODES)

            # 泊松分布生成初始最大到达率
            # np.random.poisson(lam) 用于生成一个服从泊松分布的随机数，其中 lam 是泊松分布的参数，表示单位时间内事件发生的平均次数。
            # MAX_ARRIVAL_RATE / 2.0 是为了让初始到达率在最大值的一半左右，增加环境的挑战性和动态性。
            initial_lambda = np.random.poisson(config.MAX_ARRIVAL_RATE / 2.0)

            # 将生成的流信息记录在 active_flows 列表中，供环境后续使用
            self.active_flows.append({
                'flow_id': f,
                'chain': chain,
                'start_node': start_node,
                'lambda': float(initial_lambda)
            })
    
    def _init_data_dependencies(self):
        """初始化服务间传输的依赖数据量 (MB)"""
        for flow in self.active_flows:
            chain = flow['chain']
            # 遍历链上的相邻服务节点，分配通信载荷
            for i in range(len(chain) - 1):
                s_p = chain[i]
                s_q = chain[i + 1]
                if self.data_dep_matrix[s_p, s_q] == 0:
                    # 随机分配 1~10 MB 的数据依赖量
                    self.data_dep_matrix[s_p, s_q] = np.random.uniform(1.0, config.MAX_DATA_MB)
        
    def step_traffic(self):
        """
        模拟潮汐效应，更新请求到达率。
        在仿真的不同时间步调用，使网络处于动态波动中。
        """
        for flow in self.active_flows:
            # 加入随机游走噪音
            delta = np.random.normal(0, 50.0)
            new_lambda = flow['lambda'] + delta
            # 防止到达率突破上下限
            flow['lambda'] = np.clip(new_lambda, 10.0, config.MAX_ARRIVAL_RATE)

class HeuristicRouter:
    """
    作用：快时间尺度路由决策引擎。
    完全摒弃 for 循环，利用纯 Numpy 的 Einsum/广播张量运算，
    快速推导全网服务路由概率，并计算 Burke 定理下的聚合到达率矩阵。
    """
    def __init__(self, physical_net):
        self.physical_net = physical_net
        self.alpha = config.ROUTING_ALPHA
        self.gamma = config.ROUTING_GAMMA

    def calculate_routing_probability(self, N_matrix):
        """
        计算自适应转发概率张量 P \in R^{(S, V, V)}。
        P_tensor[s, i, j] 代表：当前处于节点 i 时，为了执行服务 s，将流量转发至节点 j 的概率。
        """
        # ================= 1. 计算能力权重张量 (Capacity Preference) =================
        # 统计每个服务在全网的实例总数 N_sum，形状 (S,)
        N_sum = N_matrix.sum(axis=0)
        # 防止除零异常
        # np.where(condition, x, y) 函数的作用是根据 condition 条件返回 x 或 y 中的元素。
        N_sum_safe = np.where(N_sum == 0, 1e-9, N_sum)
        # C_matrix[j, s] 表示节点 j 拥有的服务 s 实例数占比，形状 (V, S)
        # N_matrix含义是每个节点 i 上部署的服务 s 的实例数量，形状 (V, S)
        # N _sum_safe 是每个服务在全网的实例总数，形状 (S,)
        # 通过广播机制，N_sum_safe 会自动扩展为 (1, S)，使得每个节点 j 的服务 s 实例占比正确计算。
        C_matrix = N_matrix / N_sum_safe
        # 将其转置并增加维度，升维广播至 (S, 1, V)，使得来源节点 i 的维度共享同一权重
        # 转置是为了后续计算中，服务 s 的维度能够正确对齐到第一维，方便与延迟矩阵进行加权组合。延迟矩阵的维度是 (V, V)，其中第一维是来源节点 i，第二维是目标节点 j。
        # 增加维度是为了在后续计算中，能够利用广播机制将节点 j 的能力权重正确应用到所有来源节点 i 上。
        C_tensor = C_matrix.T[:, np.newaxis, :]

        # ================= 2. 计算延迟偏好张量 (Latency Preference) =================
        # 获取物理拓扑的传播延迟矩阵 d_matrix，形状 (V, V)
        d_matrix = self.physical_net.edge_features[:, :, 1]
        # 指数衰减转换矩阵 E，距离越近值越大，形状 (V, V)
        E = np.exp(-self.gamma * d_matrix)

        # 指示变量矩阵 x_matrix：判断节点是否有对应服务的实例，形状 (V, S)
        x_matrix = (N_matrix > 0).astype(np.float32)
        # 升维广播至 (S, 1, V)
        x_tensor = x_matrix.T[:, np.newaxis, :]

        # 广播相乘：Num[s, i, j] = E[i, j] * x_tensor[s, 0, j]
        # 形状变为 (S, V, V)
        Num = E[np.newaxis, :, :] * x_tensor

        # 对目标节点 j (axis=2) 求和，计算归一化分母，形状 (S, V, 1)
        Denom = Num.sum(axis=2, keepdims=True)
        # 防止除零异常
        Denom_safe = np.where(Denom == 0, 1e-9, Denom)
        L_tensor = Num / Denom_safe # 形状 (S, V, V)

        # ================= 3. 加权融合与物理常识拦截 =================
        P_tensor = self.alpha * C_tensor + (1 - self.alpha) * L_tensor

        # 终极掩码：如果某个服务在全网没有任何实例 (N_sum == 0)，则彻底掐断通向该服务的路由概率
        valid_mask = (N_sum >0)[:, np.newaxis, np.newaxis] # 形状 (S, 1, 1)，广播至 (S, V, V)
        P_tensor *= valid_mask

        return P_tensor
    
    def step_route(self, N_matrix, traffic_generator):
        """
        执行快时间尺度的拓扑链式路由传递，输出用于后续排队计算的到达率矩阵。
        """
        # 获取底层路由概率张量
        P_tensor = self.calculate_routing_probability(N_matrix)

        # 初始化聚合到达率矩阵 lambda_agg \in R^{(V, S)}
        lambda_agg = np.zeros((config.NUM_NODES, config.NUM_SERVICES), dtype=np.float32)

        # 记录跨节点通信流量张量 F \in R^{(S, V, V)} 
        # F_tensor[s, i, j] 代表：为了执行服务 s，从节点 i 转发至节点 j 的流量强度 (req/s)，用于后续计算通信延迟和链路负载。
        F_tensor = np.zeros((config.NUM_SERVICES, config.NUM_NODES, config.NUM_NODES), dtype=np.float32)

        # 遍历每一条动态生成的调用流进行链路推导
        for flow in traffic_generator.active_flows:
            chain = flow['chain']
            start_node = flow['start_node']
            lam = flow['lambda']

            # current_lambda[i] 表示当前停留在节点 i 等待进入下一个服务的请求量
            current_lambda = np.zeros(config.NUM_NODES, dtype=np.float32)
            current_lambda[start_node] = lam

            prev_s = None
            for s in chain:
                # 核心张量乘法！基于概率矩阵，将 current_lambda 瞬间分发至目标节点
                # 矩阵运算: (1, V) @ (V, V) -> (1, V)
                next_lambda = current_lambda @ P_tensor[s]

                # 叠加到全局服务节点聚合速率矩阵中
                lambda_agg[:, s] += next_lambda

                # 若存在前驱节点，则计算跨越物理链路产生的数据交换量 F_step
                if prev_s is not None:
                    # current_lambda[:, np.newaxis] 将 (V,) 竖起变为 (V, 1)，与 (V, V) 广播相乘
                    F_step = (current_lambda[:, np.newaxis] * P_tensor[s]) # 形状 (V, V)，表示从每个来源节点 i 到每个目标节点 j 的流量强度
                    F_tensor[prev_s] += F_step

                # 流量向前推进，当前服务执行完后，驻留量变为进入下一跳的出发量
                current_lambda = next_lambda
                prev_s = s
        # 返回聚合到达率、通信流量矩阵与概率图谱
        
        return lambda_agg, F_tensor, P_tensor