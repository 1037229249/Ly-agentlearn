"""
本文件是强化学习算法交互的官方入口，继承自 Gymnasium。
定义 observation_space 和实现 reset 方法。
状态空间必须采用 Dict 形式，以完美表达多维度的网络状态（剩余资源、资源利用率等）。
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import config
from topology import PhysicalNetwork

#这里HybridOrchestrationEnv 是一个强化学习环境类，继承自 gym.Env，表示一个混合编排环境。
class HybridOrchestrationEnv(gym.Env):
    def __init__(self):
        super(HybridOrchestrationEnv, self).__init__()#调用父类的构造函数，初始化环境

        # 实例化底层的物理网络基座
        self.physical_net = PhysicalNetwork()
        self_num_nodes = config.NUM_NODES

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
        # 动作空间 (Action Space)逐步完善
        self.action_space = None # 先占位，后续根据调度决策的具体设计来定义动作空间

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
        """
        # 这里暂时占位，后续根据调度决策的具体设计来实现动作执行逻辑
        raise NotImplementedError("Step function is not implemented yet.")