"""
day1
强化学习算法交互的官方入口，继承自 Gymnasium。
定义 observation_space 和实现 reset 方法。
状态空间必须采用 Dict 形式，以完美表达多维度的网络状态（剩余资源、资源利用率等）。
day2
引入了 `action_space` 的定义，
将多实例部署决策（每个节点每个服务部署几个）扁平化为一维的 MultiDiscrete 空间。
同时在 `step()` 函数中，我们利用 `np.dot` (矩阵点乘) 瞬间计算所有节点的资源消耗。
并加入越界判定机制：一旦智能体乱部署导致资源超载，立刻给予极大负向奖励并强制结束回合。
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import config
from topology import PhysicalNetwork
from services import ServiceRegistry

#这里HybridOrchestrationEnv 是一个强化学习环境类，继承自 gym.Env，表示一个混合编排环境。
class HybridOrchestrationEnv(gym.Env):
    def __init__(self):
        super(HybridOrchestrationEnv, self).__init__()#调用父类的构造函数，初始化环境

        # 实例化底层的物理网络基座
        self.physical_net = PhysicalNetwork()
        # 实例化服务注册表，生成服务资源需求矩阵
        self.service_registry = ServiceRegistry()

        self_num_nodes = config.NUM_NODES
        self_num_services = config.NUM_SERVICES

        # ================= 状态空间 (Observation Space) =================
        # 使用 Dict 结构，Dict结构指的是状态空间由多个不同类型的子空间组成，每个子空间都有一个唯一的键来标识它。
        # 命名一个observation_space而不是其他名字，是因为这是Gym环境中预定义的属性，强化学习算法会默认访问这个属性来获取环境的状态空间定义
        # space.Dict 表示状态空间是一个字典类型，字典中的每个键对应一个不同的子空间，这些子空间可以是不同类型的空间（如 Box、Discrete 等），用于描述环境状态的不同方面。
        self.observation_space = spaces.Dict({
            # 节点的剩余可用资源：维度 (NUM_NODES, 3)，范围从 0 到 正无穷
            # spaces.Box 是 Gym 中用于定义连续空间的类，表示一个 n 维的连续空间，其中每个维度都有一个最小值和一个最大值。
            # low=0 表示每个维度的最小值为 0，high=np.inf 表示每个维度的最大值为正无穷，shape=(self_num_nodes, 3) 表示这个空间是一个二维数组
            # 第一维的大小为节点数量（NUM_NODES），第二维的大小为 3，分别对应 CPU、GPU 和内存三个资源维度，dtype=np.float32 表示这个空间中的数据类型为 32 位浮点数。
            "node_remaining_resources": spaces.Box(
                low=0, 
                high=np.inf, 
                shape=(self_num_nodes, 3), 
                dtype=np.float32
            ),
            # 节点的资源利用率：维度 (NUM_NODES, 3)，范围从 0.0 到 1.0
            "node_resource_utilization": spaces.Box(
                low=0.0, 
                high=1.0, 
                shape=(self_num_nodes, 3), 
                dtype=np.float32
            )
        })
        # ================= 动作空间 (Action Space) 定义 =================
        # 部署动作：每个节点上每种服务部署的数量。
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

        # 初始状态下，没有任何服务部署，利用率全为 0 矩阵
        initial_observation = np.zeros((config.NUM_NODES, 3), dtype=np.float32)

        # 构建符合 observation_space 定义的状态字典
        obs = {
            "node_remaining_resources": self.physical_net.remain_matrix,
            "node_resource_utilization": initial_observation
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
        truncated = False
        info = {}

        if is_invalid_action:
            # 【非法动作分支】
            reward = config.INVALID_ACTION_PENALTY  # 极大负向奖励
            terminated = True  # 强制结束回合
            # 当前使用率维持原状,np.zeros_like()函数创建一个与capacity_matrix形状相同的全零矩阵，表示资源利用率为0
            current_util = np.zeros_like(self.physical_net.capacity_matrix)
            #"violation_count" 键的值为 1，表示发生了一次违规行为，这个信息可以在训练过程中通过 Tensorboard 等工具进行监控和分析。
            info["violation_count"] = 1 # 在 info 字典中记录违规，方便日后通过 Tensorboard 监控
        else:
            # 【合法动作分支】
            # 部署有效，执行资源扣减，更新物理环境当前可用资源
            self.physical_net.remain_matrix = self.physical_net.capacity_matrix - consumed_resources
            # 计算资源利用率公式：U_v = 消耗量 / 总容量 (注意防止除 0 导致 nan，虽然容量通常不会为 0)
            # 利用 np.clip 将利用率严格限制在 0.0 - 1.0 之间，后两个参数分别是最小值和最大值，确保计算结果不超过范围。
            current_util = np.clip(consumed_resources / self.physical_net.capacity_matrix, 0.0, 1.0)

            # 此刻由于我们还没做延迟计算，合法部署暂时给 0 分
            reward = 0.0
            # 正常跑完当前 step，回合继续
            terminated = False
            info["violation_count"] = 0

        # 4. 构建下一时刻返回的观测状态 (Observation)
        obs = {
            "node_remaining_resources": np.copy(self.physical_net.remain_matrix),  # 返回当前剩余资源状态
            "node_resource_utilization": current_util
        }

        # 按照 Gymnasium 最新 API 规范，返回五元组：obs, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info