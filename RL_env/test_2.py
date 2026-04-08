"""
Day 2 验收脚本。
测试动作空间的维度，检验非法部署能否被精准拦截并施加惩罚，以及合法部署的矩阵运算是否准确无误。
"""
import numpy as np
from hybrid import HybridOrchestrationEnv
import config

def run_test():
    env = HybridOrchestrationEnv()
    obs, _ = env.reset()
    
    # 1. 验证动作空间
    action_len = config.NUM_NODES * config.NUM_SERVICES
    assert env.action_space.shape[0] == action_len, "动作空间展平维度错误"
    print("✅ 动作空间维度检验通过")

    # 2. 构造【非法动作】：让所有节点上的所有服务都拉满 (部署 MAX_INSTANCES 个)
    # 这绝对会引起边缘节点的 CPU 和 GPU 爆炸
    bad_action = np.full(action_len, config.MAX_INSTANCES, dtype=np.int32)
    
    obs_next, reward, terminated, truncated, info = env.step(bad_action)
    assert reward == config.INVALID_ACTION_PENALTY, "系统未能给超载动作施加正确的负反馈惩罚"
    assert terminated == True, "系统未能强制终止违规的回合"
    assert info.get("violation_count") == 1, "违规未被正确记录到 info 中"
    print("✅ 非法动作（资源越界）拦截检验通过")

    # 3. 构造【合法动作】：重置环境后，只在云端(索引0) 部署 1 个 Full-size AI (全局索引 config.NUM_MICROSERVICES)
    env.reset()
    good_action = np.zeros(action_len, dtype=np.int32)
    # 计算一维数组中的精确索引
    # 展平规则：[node0_svc0, node0_svc1..., node1_svc0...]
    # 所以 云节点0 的 AI 服务 0 的索引为：0 * NUM_SERVICES + NUM_MICROSERVICES
    cloud_full_ai_idx = 0 * config.NUM_SERVICES + config.NUM_MICROSERVICES
    good_action[cloud_full_ai_idx] = 1 # 部署 1 个实例
    
    obs_next, reward, terminated, _, _ = env.step(good_action)
    
    assert reward == 0.0, "合法动作暂时应返回0惩罚"
    assert terminated == False, "合法动作不应终止回合"
    
    # 手动验算：云端初始 CPU 为 1024， Full-size AI 需要 16 核，那么剩余应该是 1008
    cloud_cpu_remain = obs_next["node_remaining_resources"][0, 0]
    expected_cpu = config.CLOUD_CPU - 16.0
    assert np.isclose(cloud_cpu_remain, expected_cpu), f"矩阵资源扣减计算有误！预期 {expected_cpu}, 实际 {cloud_cpu_remain}"
    
    # 验证利用率是否更新大于 0
    cloud_cpu_util = obs_next["node_resource_utilization"][0, 0]
    assert cloud_cpu_util > 0.0, "状态中的资源利用率未能正确更新"
    
    print("✅ 合法动作的部署状态流转检验完全通过！")
    print("🎉 Day 2 里程碑达成！智能体现在懂得硬件的物理极限了。")

if __name__ == "__main__":
    run_test()