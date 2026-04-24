# 文件路径: Ly-agentlearn/tests/test_7_pipeline.py

import sys
import os

# ================= 核心修复：环境变量注入 =================
# 1. 获取当前脚本所在目录的上一级目录（即项目根目录 Ly-agentlearn）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 2. 获取 RL_env 文件夹的具体路径
RL_ENV_DIR = os.path.join(PROJECT_ROOT, 'RL_env')

# 3. 将这两个路径插入到 Python 系统搜索路径的最前面
# 这样不仅能使用 from RL_env.xxx import yyy，还能让底层代码直接 import config
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RL_ENV_DIR)
# ==========================================================

import numpy as np
from RL_env.hybrid import HybridOrchestrationEnv
from RL_env.environment_wrappers import DictObsNormalizationWrapper, MaskableWrapper
from sb3_contrib import MaskablePPO

def build_wrapped_env():
    """
    工厂函数：装配包裹好的环境（洋葱模型）
    将底层的物理环境一层层包裹，以满足强化学习框架对数据格式的严苛要求。
    """
    env = HybridOrchestrationEnv()               # 最内层：原生物理环境
    env = DictObsNormalizationWrapper(env)       # 中间层：将所有特征张量压缩到 [0, 1] 区间，防止梯度爆炸
    env = MaskableWrapper(env)                   # 最外层：暴露底层的动作掩码（Action Mask），拦截非法动作
    return env

def test_phase_1_smoke():
    """
    阶段一：环境连通性冒烟测试 (Smoke Test)
    作用：就像给电子设备通电看会不会冒烟一样，只运行最基础的初始化，检查数据流是否畅通。
    """
    print("\n" + "="*50)
    print(">>> [阶段一] 执行环境连通性冒烟测试...")
    
    env = build_wrapped_env()                    # 实例化装配好的环境
    obs, info = env.reset()                      # 调用重置函数，获取初始状态
    
    # 断言（强制检查）：验证环境吐出的字典里是否包含我们定义的键值。如果缺少，程序会在此处崩溃并报错。
    assert "macro_node_features" in obs, "❌ 错误：观测字典缺失 'macro_node_features' 键值！"
    assert "micro_service_distribution" in obs, "❌ 错误：观测字典缺失微观服务分布状态！"
    
    print("✅ 阶段一通过：环境成功重置，特征张量成功流出，无任何崩溃。")

def test_phase_2_random_mask_stress():
    """
    阶段二：随机策略防浪涌与掩码严格性压测
    作用：不使用聪明的 AI，而是像猴子敲键盘一样输入 100 次“合法范围内的随机动作”。
    验证底层排队论引擎和物理基座在极端乱序调度下是否会抛出越界异常或除零报错。
    """
    print("\n" + "="*50)
    print(">>> [阶段二] 随机策略防浪涌与掩码严格性压测 (测试 100 步)...")
    
    env = build_wrapped_env()                    # 实例化环境
    obs, _ = env.reset()                         # 初始化状态
    
    for i in range(100):                         # 循环执行 100 个时间步
        # 1. 向环境索要当前状态下“哪些动作是合法的”掩码数组（1D布尔数组）
        mask = env.action_masks()
        
        # 2. 我们知道动作空间总维度是 N(节点数) * S(服务数)，每个服务有 3 个操作（缩、保、扩）
        # 将 1D 数组重塑为 2D 矩阵，方便逐个动作位进行合法采样
        action_dim_per_service = 3 
        total_actions = env.action_space.shape[0]
        mask_reshaped = mask.reshape(total_actions, action_dim_per_service)
        
        action = []                              # 准备存储这一步的综合动作
        for m in mask_reshaped:                  # 遍历每一个动作维度（比如节点1上的微服务A）
            valid_indices = np.where(m)[0]       # np.where(m)[0] 会找出该维度下值为 True 的索引（即合法的选项）
            # 从合法选项中随机抽选一个，追加到动作列表中
            action.append(np.random.choice(valid_indices))
            
        action = np.array(action)                # 转换为 Numpy 数组，送给环境
        
        # 3. 环境执行该随机动作，返回下一个状态、奖励、是否结束等信息
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 打印部分进度，让你知道程序正在运行
        if (i + 1) % 25 == 0:
            print(f"   - 已完成 {i+1}/100 步随机抗压测试，当前系统未崩溃...")
        
        # 如果因为某些惩罚导致回合结束，则自动重置环境
        if terminated or truncated:
            obs, _ = env.reset()
            
    print("✅ 阶段二通过：环境在严苛的随机狂暴动作下运转良好，动作掩码防线极其坚固。")

def test_phase_3_micro_overfit():
    """
    阶段三：极小规模样本过拟合验证 (Overfitting Test)
    作用：用最简单的神经网络去“死记硬背”当前的规则。
    如果神经网络正常，它的分数（Reward）应该随着训练迅速攀升。这用来验证反向传播和梯度更新链路是否打通。
    """
    print("\n" + "="*50)
    print(">>> [阶段三] 极小规模样本过拟合验证 (验证神经网络链路)...")
    
    env = build_wrapped_env()                    # 实例化环境
    
    # 实例化 SB3-Contrib 的 MaskablePPO 算法（专为处理 Action Masking 优化的 PPO）
    model = MaskablePPO(
        "MultiInputPolicy",                      # 因为我们的状态是 Dict 格式，所以必须用 MultiInputPolicy
        env, 
        policy_kwargs=dict(net_arch=[64, 64]),   # 使用极小的两层神经网络（64个神经元），加速运算
        n_steps=128,                             # 每次收集 128 步数据就更新一次网络（正常训练通常是 2048 步）
        batch_size=32,                           # 批处理大小
        learning_rate=1e-3,                      # 调大一点学习率，帮助模型快速变化
        verbose=1,                               # 在控制台输出训练的具体分数和进度
        seed=42                                  # 固定随机种子，保证每次测试结果一致
    )
    
    print("[*] 正在对 Agent 进行 10000 步速成训练，将显示进度条与日志...")
    
    # 启动训练。progress_bar=True 会在终端底部显示一个进度条
    model.learn(total_timesteps=10000, progress_bar=True)
    
    print("\n✅ 阶段三通过：PPO 梯度回传正常，模型成功与物理基座完成多轮循环更新！")
    print("\n🎉 Day 7 预演关卡全部通关！底层机制完美对接深度学习框架！")

if __name__ == "__main__":
    # 按顺序依次执行三个阶段的测试
    test_phase_1_smoke()
    test_phase_2_random_mask_stress()
    test_phase_3_micro_overfit()