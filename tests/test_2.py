# 文件路径: Ly-agentlearn/RL_env/test_3_action_mask.py
"""
文件作用：新版 Day 2 动作掩码 (Action Masking) 验收脚本。
目标：验证环境能够基于物理拓扑状态，动态且精确地拦截所有违背《指南》架构约束的非法动作选项。
"""
import numpy as np
from hybrid import HybridOrchestrationEnv
import config

def run_test():
    env = HybridOrchestrationEnv()
    obs, _ = env.reset()
    
    print(">>> 正在执行新版 Day 2：混合决策掩码 (Action Masking) 验证...\n")

    # 提取掩码并还原为 (Nodes, Services, Action_Dim) 的三维形态便于断言
    flat_mask = env.action_masks()
    mask_3d = flat_mask.reshape((config.NUM_NODES, config.NUM_SERVICES, config.DEPLOY_ACTION_DIM))

    # =====================================================================
    # 测试点 1：初始状态下，没有任何实例，因此所有服务的【缩容】动作掩码必须为 False
    # =====================================================================
    assert np.all(mask_3d[:, :, 0] == False), "❌ 初始状态下出现了非法的缩容动作概率！"
    print("✅ 初始状态【缩容锁死】拦截逻辑验证通过。")

    # =====================================================================
    # 测试点 2：架构强制约束。Full-size AI (全局索引 20) 在边缘节点 (索引 1~9) 的扩容必须被屏蔽
    # =====================================================================
    full_ai_idx = config.NUM_MICROSERVICES
    # 检查节点 1 的 Full-size AI 扩容掩码 (动作2)
    edge_full_ai_expand_mask = mask_3d[1, full_ai_idx, 2]
    assert edge_full_ai_expand_mask == False, "❌ 边缘节点竟然允许部署强制云端的大模型！"
    
    # 云端 (节点 0) 资源充沛，应该允许部署
    cloud_full_ai_expand_mask = mask_3d[0, full_ai_idx, 2]
    assert cloud_full_ai_expand_mask == True, "❌ 云端正常的大模型部署被意外屏蔽了！"
    print("✅ Full-size AI 云边物理架构限制 (Roofline/显存瓶颈) 屏蔽逻辑验证通过。")

    # =====================================================================
    # 测试点 3：资源耗尽逼近测试 (极端算力约束)
    # =====================================================================
    # 人为将边缘节点 2 的 GPU 余量榨干（设为 0）
    env.physical_net.remain_matrix[2, 1] = 0.0  
    
    # 重新生成掩码
    new_mask_3d = env.action_masks().reshape((config.NUM_NODES, config.NUM_SERVICES, config.DEPLOY_ACTION_DIM))
    
    # 假设 AI 服务 1 (Light-weight AI) 需要至少 1 张 GPU (这取决于 services.py 中的配置)
    light_ai_idx = config.NUM_MICROSERVICES + 1
    gpu_depleted_mask = new_mask_3d[2, light_ai_idx, 2]
    
    assert gpu_depleted_mask == False, "❌ 节点 GPU 已耗尽，却未屏蔽 GPU 依赖的 AI 服务扩容指令！"
    print("✅ GPU / CPU 极端资源耗尽逼近检测通过。")

    print("\n🎉 新版 Day 2 里程碑达成！智能体现在通过掩码被关在了 '物理定律的铁笼' 里，再也不会因为越界引发仿真器爆炸了！")

if __name__ == "__main__":
    run_test()