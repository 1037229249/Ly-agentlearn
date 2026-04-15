"""
文件作用：Gymnasium 环境观测空间归一化与掩码协议封装层 (Wrappers)。
功能：
1. 消除各物理维度的量纲差异（如带宽的 Mbps 与 实例数的个），将所有状态矩阵压缩到 [0, 1] 紧凑区间。
2. 桥接 Stable-Baselines3-Contrib 的 ActionMasker 接口，暴露底层环境的 action_masks 方法。
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import config

class DictObsNormalizationWrapper(gym.ObservationWrapper):
    """
    异构字典观测状态归一化 Wrapper。
    核心逻辑：针对 Dict 观测空间内的矩阵分别进行静态极值归一化 (Min-Max Scaling)。
    """
    def __init__(self, env):
        super().__init__(env)
        # 获取原始环境的 observation_space, 这里observation_space 是一个 Dict 类型，包含多个键，每个键对应一个空间
        old_space = env.observation_space

        # 重构 observation_space
        # 原始的 micro_service_distribution 数据类型为 np.int32，值域为 [0, MAX_INSTANCES]
        # 经过归一化除法后，它会变成 np.float32 且值域变为 [0.0, 1.0]，必须显式更新 Space 定义，否则 Gym 的数据校验会报错
        self.observation_space = spaces.Dict({
            "macro_node_features": old_space["macro_node_features"],  # 利用率原本就在 [0, 1] 之间，保持不变
            "macro_edge_features": old_space["macro_edge_features"],  # 待在 observation 方法中限制边界
            # shape指的是 micro_service_distribution 的矩阵维度，dtype 指定为 np.float32，low 和 high 分别指定了归一化后的值域范围
            "micro_service_distribution": spaces.Box(
                low=0.0, high=1.0, shape=old_space["micro_service_distribution"].shape, dtype=np.float32
            )
        })
    
    def observation(self, obs):
        """
        拦截底层环境 step() 或 reset() 抛出的原始 obs，处理后再送入神经网络。
        """
        # 1. 宏观节点特征：已经是 0-1 的利用率，直接透传
        node_features = obs["macro_node_features"]

        # 2. 微观服务分布：[0, MAX_INSTANCES] -> [0.0, 1.0]
        # 使用 astype 转换为 float32 防止整数除法引发的精度丢失
        micro_dist = obs["micro_service_distribution"].astype(np.float32) / config.MAX_INSTANCES

        # 3. 拓扑边特征：由于包含无穷大 (np.inf) 和绝对数值，需彻底清洗
        edge_features = obs["macro_edge_features"].copy()  # 先复制一份，避免修改原始环境的状态

        # ---- 处理通道 0：带宽 (Bandwidth) ----
        # 物理拓扑中，节点对自身的内部带宽为 np.inf。如果不剔除，会导致网络输出 NaN。
        # 这里将其替换为 MAX_NORM_BANDWIDTH，这样归一化后就代表最高质量的链路 (1.0)
        # where 函数根据条件 np.isinf(edge_features[:, :, 0]) 返回一个布尔数组,
        # 这个条件检查 edge_features 的第一个通道（带宽）中哪些元素是无穷大。
        # 在该数组中为 True 的位置将 edge_features[:, :, 0] 替换为 config.MAX_NORM_BANDWIDTH，否则保持原值不变。
        edge_features[:, :, 0] = np.where(
            np.isinf(edge_features[:, :, 0]), 
            config.MAX_NORM_BANDWIDTH, 
            edge_features[:, :, 0]
        )
        # 极值除法归一化，并用 clip 裁剪以防万一超过 1.0 的扰动
        edge_features[:, :, 0] = np.clip(edge_features[:, :, 0] / config.MAX_NORM_BANDWIDTH, 0.0, 1.0)

        # ---- 处理通道 1：延迟 (Latency) ----
        # 物理拓扑中，云端到边缘的延迟可能达到 50ms。将其按设定上限进行归一化。
        edge_features[:, :, 1] = np.clip(edge_features[:, :, 1] / config.MAX_NORM_LATENCY, 0.0, 1.0)

        # 返回清洗后、纯洁的 [0, 1] 字典空间给智能体
        return {
            "macro_node_features": node_features,
            "macro_edge_features": edge_features,
            "micro_service_distribution": micro_dist
        }
    
class MaskableWrapper(gym.Wrapper):
    """
    违规动作屏蔽 Wrapper 协议桥接。
    作用：为 SB3-Contrib 的 MaskablePPO 提供标准调用接口。
    """
    def action_masks(self):
        """
        拦截智能体获取掩码的请求，并将其透传（Pass-through）给被包裹的最底层环境（unwrapped）。
        """
        return self.env.unwrapped.action_masks()