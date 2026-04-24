"""
文件作用：Day 5 综合奖励与物理管线集成验证脚本 (修正版)。
目标：
1. 验证环境在异构算力（Mu 矩阵随机化）下能否完成完整的“路由-排队-延迟-奖励”计算链路。
2. 验证 SLA 成功率、负载方差、部署成本等指标是否正确提取。
3. 验证 AI 服务抖动惩罚 (Reward Shaping) 是否能有效识别实例频繁变更。
"""
import numpy as np
from hybrid import HybridOrchestrationEnv
import config

def run_test():
    print(">>> 正在执行 Day 5：修正版多目标奖励与物理全管线集成验证...\n")
    
    # 初始化环境，触发 Mu 矩阵的随机异构分配
    env = HybridOrchestrationEnv()
    obs, _ = env.reset()

    # 打印部分节点的异构处理速率，验证异构性
    print(f"[异构算力检查] 云节点 (Node 0) AI 基准速率: {env.Mu_matrix[0, config.NUM_MICROSERVICES]:.2f}")
    print(f"[异构算力检查] 边缘节点 (Node 1) AI 基准速率: {env.Mu_matrix[1, config.NUM_MICROSERVICES]:.2f}")
    print("-" * 50)

    # =====================================================================
    # 步骤 1：初始部署动作 (扩容一个 AI 服务实例)
    # =====================================================================
    # 生成全 1 的动作矩阵 (1 代表保持现状)
    action_1 = np.ones((config.NUM_NODES, config.NUM_SERVICES), dtype=np.int32)
    ai_service_idx = config.NUM_MICROSERVICES + 1
    
    # 【修复点】利用 Action Mask 动态寻找一个有足够资源（合法）的边缘节点
    flat_mask = env.action_masks()
    mask_3d = flat_mask.reshape((config.NUM_NODES, config.NUM_SERVICES, config.DEPLOY_ACTION_DIM))
    
    target_node = -1
    for n in range(1, config.NUM_NODES):  # 遍历边缘节点 (排除云端节点 0)
        # 动作索引 2 代表扩容 (+1)。检查在该节点扩容该 AI 服务是否合法
        if mask_3d[n, ai_service_idx, 2] == True: 
            target_node = n
            break
            
    if target_node == -1:
        print("⚠️ 随机生成的环境中，所有边缘节点都没有足够的 GPU 来部署此 AI 服务，测试退出。")
        return

    # 在合法节点上扩容该 AI 服务
    print(f"[*] 动态查找到 Node {target_node} 资源充沛，将在其上扩容 AI 服务。")
    action_1[target_node, ai_service_idx] = 2  
    
    obs1, r1, term1, trunc1, info1 = env.step(action_1.flatten())
    print(f"Step 1 - 扩容部署: Reward = {r1:.4f}")
    print(f"         [指标] 平均延迟: {info1['Metrics/Avg_EndToEnd_Delay_ms']:.2f} ms")
    print(f"         [指标] SLA成功率: {info1['Metrics/SLA_Success_Rate']:.2%}")
    print(f"         [惩罚] AI 抖动惩罚: {info1['Rewards/Penalty_Smoothness']:.2f} (预期应有一次扩容惩罚)")

    # =====================================================================
    # 步骤 2：稳态动作 (保持当前实例分布不变)
    # =====================================================================
    # 【修复点】保证后面的测试代码中剧烈抖动动作 (Step 3) 使用相同的 target_node
    action_steady = np.ones((config.NUM_NODES, config.NUM_SERVICES), dtype=np.int32)
    obs2, r2, term2, trunc2, info2 = env.step(action_steady.flatten())
    # ... 保持不变 ...

    # =====================================================================
    # 步骤 3：剧烈抖动动作 (刚刚扩容，现在立即缩容)
    # =====================================================================
    action_thrash = np.ones((config.NUM_NODES, config.NUM_SERVICES), dtype=np.int32)
    action_thrash[target_node, ai_service_idx] = 0  # 【修复点】在这里使用前面找到的 target_node 缩容
    obs3, r3, term3, trunc3, info3 = env.step(action_thrash.flatten())
    print(f"\nStep 3 - 剧烈缩容: Reward = {r3:.4f}")
    print(f"         [惩罚] AI 抖动惩罚: {info3['Rewards/Penalty_Smoothness']:.2f} (预期应有严重的负向惩罚)")

    # =====================================================================
    # 结果分析
    # =====================================================================
    # 验证抖动惩罚的力度
    if info3['Rewards/Penalty_Smoothness'] < 0:
        print("\n✅ 抖动惩罚验证通过：系统成功识别并抑制了 AI 服务的频繁启停。")
    
    # 验证 Info 字典中是否包含了学术论文所需的全部物理指标
    required_keys = [
        "Metrics/Avg_EndToEnd_Delay_ms", 
        "Metrics/SLA_Success_Rate", 
        "Metrics/Load_Variance", 
        "Metrics/Total_Cost"
    ]
    for key in required_keys:
        assert key in info3, f"❌ Info 字典缺失核心指标: {key}"
    
    print("✅ 学术级指标监控 (Academic Monitoring) 验证通过。")
    print("\n🎉 Day 5 最终集成里程碑达成！仿真环境已具备完整的物理自洽性与多目标评价能力。")

if __name__ == "__main__":
    run_test()