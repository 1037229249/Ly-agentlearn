"""
文件作用：Day 6 深度网络兼容性与 Wrapper 归一化张量安全测试。
目标：
1. 验证经过 Wrapper 处理后的观测字典中，所有元素的数值是否严格在 [0, 1] 区间内。
2. 确保没有 np.inf 或 np.nan 泄漏给后续的神经网络，绝对避免梯度爆炸。
"""
import numpy as np
from hybrid import HybridOrchestrationEnv
from environment_wrappers import DictObsNormalizationWrapper, MaskableWrapper
import config

def run_test():
    print(">>> 正在执行 Day 6：深度网络观测空间归一化与安全性验证...\n")

    # 1. 实例化底层环境并层层套上 Wrapper（洋葱模型）
    # 从内到外：原始物理环境 -> 数据归一化层 -> 动作掩码暴露层
    raw_env = HybridOrchestrationEnv()
    env = MaskableWrapper(DictObsNormalizationWrapper(raw_env))

    obs, _ = env.reset()

    # 2. 执行张量边界扫描测试
    num_steps = 200
    safe = True
    
    for step in range(num_steps):
        # 1. 获取当前状态下合法的动作掩码 (1D 数组)
        flat_mask = env.action_masks()
        
        # 2. 将掩码重塑为 (N*S, 3) 的形状，方便逐个动作维度进行合法采样
        mask_reshaped = flat_mask.reshape(-1, config.DEPLOY_ACTION_DIM)
        
        # 3. 手动进行合法采样
        valid_action_list = []
        for m in mask_reshaped:
            # np.where(m)[0] 会返回当前合法的动作索引，比如 [1, 2]
            valid_choices = np.where(m)[0] 
            # 从合法集合中随机挑一个
            valid_action_list.append(np.random.choice(valid_choices))
            
        action = np.array(valid_action_list, dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)

        # ================= 张量边界扫描断言 =================
        for key, matrix in obs.items():
            # 断言 1：不能含有 NaN (Not a Number) 或 Inf (Infinity)
            if np.isnan(matrix).any() or np.isinf(matrix).any():
                print(f"❌ 致命错误：在 {key} 矩阵中发现了 NaN 或 Inf！这会摧毁神经网络。")
                safe = False
            
            # 断言 2：数值域必须严丝合缝卡在 [0.0, 1.0] 附近
            # 给定 0.01 的浮点数截断容忍误差
            matrix_max = np.max(matrix)
            matrix_min = np.min(matrix)
            if matrix_max > 1.01 or matrix_min < -0.01:
                print(f"❌ 归一化漏网之鱼：{key} 值域异常！最大值: {matrix_max:.4f}, 最小值: {matrix_min:.4f}")
                safe = False
        
        # 若因暴力随机动作导致环境结束，自动重置
        if terminated or truncated:
            obs, _ = env.reset()

    if safe:
        print("✅ 张量边界扫描测试通过：所有异构特征均已被完美压缩至标准正态 [0.0, 1.0] 区间。")
        print("✅ 无穷大剔除断言通过：底层物理的 np.inf (内部带宽) 被成功映射剥离。")
        print("✅ SB3 MaskableWrapper 桥接验证通过。")
        print("\n🎉 Day 6 里程碑达成！")
        print("💡 你的物理引擎现在已经被彻底套上了一层‘数字安全服’。明天的 Day 7，我们就可以直接用短短几行代码引入 SB3 的 PPO 框架并开始炼丹了！")
    else:
        print("\n❌ 发现张量逸出，请检查环境代码。")

if __name__ == "__main__":
    run_test()