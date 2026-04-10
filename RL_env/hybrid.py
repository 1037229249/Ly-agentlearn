"""
文件作用：基于 Gymnasium 的强化学习环境主入口（双时间尺度架构的 Macro-step 层）。
核心改造点：
1. 彻底摒弃一维扁平状态，采用 Dict 异构张量空间。
2. 动作空间改为相对增减指令（-1, 0, +1）。
3. 引入 Action Masking (动作掩码) 机制，从底层屏蔽资源越界与非法架构部署，终结暴力试错。
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import config
from topology_graph import PhysicalNetworkGraph
from services import ServiceRegistry

#这里HybridOrchestrationEnv 是一个强化学习环境类，继承自 gym.Env，表示一个混合编排环境。
class HybridOrchestrationEnv(gym.Env):
    def __init__(self):
        super(HybridOrchestrationEnv, self).__init__()#调用父类的构造函数，初始化环境

        # 实例化图拓扑物理基座与服务属性注册表
        self.physical_net = PhysicalNetworkGraph()
        # 生成服务资源需求矩阵
        self.service_registry = ServiceRegistry()

        self_num_nodes = config.NUM_NODES
        self_num_services = config.NUM_SERVICES

        # ================= 状态空间 (Observation Space) =================
        # 使用 Dict 结构，Dict结构指的是状态空间由多个不同类型的子空间组成，每个子空间都有一个唯一的键来标识它。
        # 命名一个observation_space而不是其他名字，是因为这是Gym环境中预定义的属性，强化学习算法会默认访问这个属性来获取环境的状态空间定义
        # space.Dict 表示状态空间是一个字典类型，字典中的每个键对应一个不同的子空间，这些子空间可以是不同类型的空间（如 Box、Discrete 等），用于描述环境状态的不同方面。
        self.observation_space = spaces.Dict({
            # spaces.Box 是 Gym 中用于定义连续空间的类，表示一个 n 维的连续空间，其中每个维度都有一个最小值和一个最大值。
            # low=0 表示每个维度的最小值为 0，high=np.inf 表示每个维度的最大值为正无穷，shape=(self_num_nodes, 3) 表示这个空间是一个二维数组
            # 第一维的大小为节点数量（NUM_NODES），第二维的大小为 3，分别对应 CPU、GPU 和内存三个资源维度，dtype=np.float32 表示这个空间中的数据类型为 32 位浮点数。
            # 宏观节点特征：归一化后的资源利用率 (NUM_NODES, 3)
            "macro_node_features": spaces.Box(
                low=0, high=1.0, 
                shape=(config.NUM_NODES, 3), 
                dtype=np.float32
            ),
            # 宏观边特征：包含带宽与延迟的双通道张量 (NUM_NODES, NUM_NODES, 2)
            "macro_edge_features": spaces.Box(
                low=0.0, 
                high=np.inf, 
                shape=(config.NUM_NODES, config.NUM_NODES, 2), 
                dtype=np.float32
            ),
            # 微观服务分布：精准记录各类服务在不同节点上的实例数分布矩阵 (NUM_NODES, NUM_SERVICES)
            "micro_service_distribution": spaces.Box(
                low=0, 
                high=config.MAX_INSTANCES, 
                shape=(config.NUM_NODES, config.NUM_SERVICES), 
                dtype=np.int32
            )
        })
        # ================= 动作空间 (Action Space) 定义 =================
        # 动作空间解耦为离散的增减指令 (0: 缩容, 1: 保持, 2: 扩容)
        # action_dim 的计算方式是每个节点（NUM_NODES）对每个服务（NUM_SERVICES）都可以执行一个动作
        # 因此总的动作维度是 NUM_NODES * NUM_SERVICES。
        action_dim = self_num_nodes * self_num_services
        # spaces.MultiDiscrete 是 Gym 中用于定义多离散空间的类，表示一个由多个离散空间组成的空间，每个离散空间都有自己的取值范围。
        # np.full作用创建一个长度为 action_dim 的数组，数组中的每个元素都被设置为 config.DEPLOY_ACTION_DIM（即 3）。
        self.action_space = spaces.MultiDiscrete(
            np.full(action_dim, config.DEPLOY_ACTION_DIM, dtype=np.int32)
        )

        # 用于跟踪当前服务部署情况的内部状态变量，初始为全零矩阵，维度为 (NUM_NODES, NUM_SERVICES)
        self.current_distribution = np.zeros((config.NUM_NODES, config.NUM_SERVICES), dtype=np.int32)

    def reset(self, seed=None, options=None):
        """
        这里reset方法是Gym环境中的一个重要方法，用于将环境重置到初始状态，准备开始一个新的Episode。
        参数seed用于设置随机数生成器的种子，以确保环境的可重复性，options参数可以包含一些额外的选项来定制环境的重置行为。
        Gymnasium 新版 API 要求 reset 方法返回一个包含初始观察值和额外信息的元组 (obs, info)
        其中 obs 是环境的初始状态，info 是一个字典，可以包含一些额外的信息供算法使用。
        """
        # Gymnasium 要求的随机种子初始化处理
        super().reset(seed=seed)

        # 重置底层物理拓扑，生成新的异构资源和链路状态
        self.physical_net.reset_topology()
        # 清空全网所有的实例部署
        self.current_distribution.fill(0)

        # 构建符合 observation_space 定义的状态字典
        obs = {
            "macro_node_features": self.physical_net.get_utilization_matrix(),  # 当前节点资源利用率矩阵 (NUM_NODES, 3)
            "macro_edge_features": np.copy(self.physical_net.edge_features),  # 当前边
            "micro_service_distribution": np.copy(self.current_distribution)  # 当前服务分布矩阵 (NUM_NODES, NUM_SERVICES)
        }
        # 返回状态 obs 和一个空的 info 字典（预留给后续记录方差、成本等额外指标）
        info = {}
        return obs, info
    
    def action_masks(self):
        """
        动态掩码生成器
        作用：为 Stable-Baselines3-Contrib 的 MaskablePPO 提供接口。
        返回：一维布尔数组，屏蔽物理上绝对不可行的动作，让神经网络 100% 只在合法域内探索。
        """
        # 初始化掩码：维度为 (节点数, 服务数, 3种动作)，默认全部合法 (True)
        # np.ones 函数的作用是创建一个指定形状的数组，并用 1 填充。
        # 这里创建了一个形状为 (NUM_NODES, NUM_SERVICES, DEPLOY_ACTION_DIM) 的三维数组，所有元素都被初始化为 1（即 True），表示初始状态下所有动作都是合法的。
        mask = np.ones((config.NUM_NODES, config.NUM_SERVICES, config.DEPLOY_ACTION_DIM), dtype=bool)
        # 遍历每个节点和服务，检查每种动作的合法性
        for n in range(config.NUM_NODES):
            for s in range(config.NUM_SERVICES):
                current_instances = self.current_distribution[n, s]
                req = self.service_registry.service_req_matrix[s]  # 获取服务 s 的资源需求向量 [CPU, GPU, MEM]
                # ----- 动作 0: 缩容 (-1) -----
                # 限制：如果当前实例数小于等于 0，则禁止再缩容
                if current_instances <= 0:
                    mask[n, s, 0] = False  # 禁止缩容动作
                # ----- 动作 1: 保持 (0) -----
                # 限制：保持动作通常总是合法的，因为它不改变部署状态，所以这里不做特殊处理，保持为 True。
                # ----- 动作 2: 扩容 (+1) -----
                # 限制 A: 如果当前实例数已经达到单节点最大实例数上限，则禁止扩容
                if current_instances >= config.MAX_INSTANCES:
                    mask[n, s, 2] = False  # 禁止扩容动作
                else:
                    # 限制 B: 物理剩余资源必须能够容纳该服务单实例的资源需求
                    if np.any(self.physical_net.remain_matrix[n] < req):
                        mask[n, s, 2] = False  # 禁止扩容动作
                    # 限制 C：架构限制(例如 Full-size AI 只能部署在云端 Node 0)
                    # 假设 config.NUM_MICROSERVICES 对应的是第一个 AI 服务 (Full-size)
                    if s == config.NUM_MICROSERVICES and n != 0:
                        mask[n, s, 2] = False  # 禁止在非云节点扩容 Full-size AI 服务

        # SB3 的 MaskablePPO 针对 MultiDiscrete 要求返回展开的 1D 布尔数组
        # mask 的原始形状是 (NUM_NODES, NUM_SERVICES, DEPLOY_ACTION_DIM)
        # flatten() 方法将这个三维数组展平为一维数组，长度为 NUM_NODES * NUM_SERVICES * DEPLOY_ACTION_DIM
        # 每个元素对应一个具体的动作是否合法。
        return mask.flatten()

    def step(self, action):
        """
        双时间尺度中的宏观时间步 (Macro-step Deploy)
        """
        # 1. 动作解码：将 0/1/2 映射为真实的实例变动量 -1/0/+1
        # action 是长度为 N*S 的 1D 数组，转为 (N, S) 矩阵，再整体减 1
        action_matrix = action.reshape((config.NUM_NODES, config.NUM_SERVICES))
        delta_matrix = action_matrix - 1  # 将 0/1/2 转换为 -1/0/+1 的增减指令
        
        # 2. 计算转移后的微观服务分布状态
        N_matrix = self.current_distribution + delta_matrix  # 计算新的实例分布矩阵

        # [安全网] 理论上有 Action Masking 不应发生越界，出现越界要报异常
        if np.any(N_matrix < 0) or np.any(N_matrix > config.MAX_INSTANCES):
            raise ValueError("Action leads to invalid instance count! Check action masking logic.")
        
        N_matrix = np.clip(N_matrix, 0, config.MAX_INSTANCES)  # 确保实例数在合法范围内

        # 3. 矩阵化资源计算 (运用 Einsum 或 dot 实现高维张量乘法)
        # N_matrix (Nodes, Services) 点乘 service_req_matrix (Services, 3) -> (Nodes, 3)
        # 具体理解：N_matrix 中的每个元素 N_matrix[n, s] 表示节点 n 上服务 s 的实例数量，service_req_matrix 中的每行表示一个服务的资源需求向量 [CPU, GPU, MEM]。
        consumed_resources = np.dot(N_matrix, self.service_registry.service_req_matrix) 

        # 硬件约束严密校验
        is_invalid_action = np.any(consumed_resources > self.physical_net.remain_matrix)

        reward = 0
        terminated = False
        info = {}

        if is_invalid_action:
            # 在有 Mask 的情况下，这个分支理论上不应该被触达，如果触达说明 Mask 机制失效了，直接报错暴露问题。
            reward = config.INVALID_ACTION_PENALTY
            terminated = True
            info["violation_count"] = 1
            info["error_msg"] = "Mask bypass detected!"
        else:
            # 状态合法，正式更新环境状态
            # remain_matrix等于
            self.physical_net.remain_matrix = self.physical_net.capacity_matrix - consumed_resources  # 更新剩余资源矩阵
            self.current_distribution = np.copy(N_matrix)  # 更新当前服务分布状态
            info["violation_count"] = 0
        
        obs = {
            "macro_node_features": self.physical_net.get_utilization_matrix(),  # 更新后的节点资源利用率矩阵
            "macro_edge_features": np.copy(self.physical_net.edge_features),  # 边特征保持不变
            "micro_service_distribution": np.copy(self.current_distribution)  # 更新后的服务分布矩阵
        }

        # 按照 Gymnasium 最新 API 规范，返回五元组：obs, reward, terminated, truncated, info
        # 其中 truncated 是一个布尔值，表示回合是否由于达到最大步数限制而被截断，这里我们暂时设为 False，因为我们还没有实现步数限制的逻辑。
        return obs, reward, terminated, False, info