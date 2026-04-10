"""
文件作用：基于 Gymnasium 的强化学习环境主入口。
核心改造点：彻底摒弃一维扁平状态，定义结构化的 Dict Observation Space，
分离宏观资源拓扑状态（支持 GNN）与微观服务分布状态。
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
                low=0, 
                high=np.inf, 
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
        # 每一步智能体具体的部署动作是：一个长度为 NUM_NODES * NUM_SERVICES 的一维整数数组，表示每个服务在每个节点上部署的实例数量。
        # 矩阵原始维度应为 (NUM_NODES, NUM_SERVICES)，但 Gym 的 MultiDiscrete 要求传入一维数组。
        # 因此我们把容量为 (MAX_INSTANCES + 1) 的数组重复 NUM_NODES * NUM_SERVICES 次。
        # 比如MAX_INSTANCES=5，则每个维度的可选离散动作是 0,1,2,3,4,5 (即容量为6)
        action_dim = self_num_nodes * self_num_services
        # spaces.MultiDiscrete 接受一个数组，数组每个元素代表该维度允许的离散选择数量
        # np.full()函数用于创建一个指定形状的数组，并用指定的值填充这个数组。
        # 创建了一个长度为 action_dim 的一维数组，每个元素的值都是 (config.MAX_INSTANCES + 1)
        # 表示每个服务在每个节点上可以部署的实例数量从 0 到 MAX_INSTANCES。
        self.action_space = spaces.MultiDiscrete(
            np.full(action_dim, config.MAX_INSTANCES + 1, dtype=np.int32)
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
    
    def step(self, action):
        """
        这里的 step 方法是 Gym 环境中的另一个核心方法，用于执行一个动作并返回环境的下一个状态、奖励、是否结束以及额外信息。
        参数 action 是智能体选择的动作，通常是一个表示调度决策的向量或矩阵。
        这个方法需要根据 action 来更新环境的状态，并计算奖励和是否结束的标志。
        核心步骤：接收动作 -> 形状重塑 -> 矩阵运算算消耗 -> 检查约束 -> 状态转移
        """
        # 1. 动作重塑：将一维长度为 N*S 的动作数组，恢复为 (NUM_NODES, NUM_SERVICES) 的二维矩阵
        N_matrix = action.reshape((config.NUM_NODES, config.NUM_SERVICES))

        # 2. 矩阵化资源计算
        # N_matrix 维度 (Nodes, Services) 点乘 service_req_matrix 维度 (Services, 3) 
        # 结果维度必为 (Nodes, 3)，代表每个物理节点在 [CPU, GPU, MEM] 上的综合消耗量
        # mp.dot()函数用于执行矩阵乘法，计算每个节点的资源消耗总量。
        # N_matrix表示每个节点上部署的服务实例数量，service_req_matrix表示每个服务实例的资源需求。
        consumed_resources = np.dot(N_matrix, self.service_registry.service_req_matrix)

        # 3. 硬件约束校验：判断任何节点的消耗是否大过了它的最大物理容量
        # (使用 np.any() 如果有任何一个 True，说明存在越界)
        is_invalid_action = np.any(consumed_resources > self.physical_net.capacity_matrix)

        # 设定强化学习的默认变量
        reward = 0.0
        terminated = False
        info = {}

        if is_invalid_action:
            # 【非法动作分支】
            reward = config.INVALID_ACTION_PENALTY  # 极大负向奖励
            terminated = True  # 强制结束回合
            #"violation_count" 键的值为 1，表示发生了一次违规行为，这个信息可以在训练过程中通过 Tensorboard 等工具进行监控和分析。
            info["violation_count"] = 1 # 在 info 字典中记录违规，方便日后通过 Tensorboard 监控
        else:
            # 【合法动作分支】
            # 部署有效，执行资源扣减，更新物理环境当前可用资源
            self.physical_net.remain_matrix = self.physical_net.capacity_matrix - consumed_resources
            # 当前current_distribution 直接更新为 N_matrix，表示当前的服务部署状态完全由智能体的动作决定
            self.current_distribution = np.copy(N_matrix)

            # 此刻由于我们还没做延迟计算，合法部署暂时给 0 分
            reward = 0.0
            # 正常跑完当前 step，回合继续
            terminated = False
            info["violation_count"] = 0

        # 4. 构建下一时刻返回的观测状态 (Observation)
        obs = {
            "macro_node_features": self.physical_net.get_utilization_matrix(),  # 当前节点资源利用率矩阵 (NUM_NODES, 3)
            "macro_edge_features": np.copy(self.physical_net.edge_features),  # 当前边
            "micro_service_distribution": np.copy(self.current_distribution)  # 当前服务分布矩阵 (NUM_NODES, NUM_SERVICES)
        }

        # 按照 Gymnasium 最新 API 规范，返回五元组：obs, reward, terminated, truncated, info
        # 其中 truncated 是一个布尔值，表示回合是否由于达到最大步数限制而被截断，这里我们暂时设为 False，因为我们还没有实现步数限制的逻辑。
        return obs, reward, terminated, False, info