"""
文件作用：Day 3 快时间尺度服务图谱解析与实例感知自适应路由验收脚本。
目标：验证拓扑约束（尾端必为AI服务）、流量总量守恒，以及自适应权重在节点无实例时的合法性过滤。
"""
import numpy as np
from traffic_routing import DynamicTrafficGenerator, HeuristicRouter
from topology_graph import PhysicalNetworkGraph
import config

def run_test():
    print(">>> 正在执行 Day 3：快时间尺度服务路由与拓扑约束验证...\n")

    # ================= 1. 拓扑图谱校验 =================
    generator = DynamicTrafficGenerator()
    print(f"随机生成了 {len(generator.active_flows)} 条独立调用链。")
    for i, flow in enumerate(generator.active_flows):
        tail_service = flow['chain'][-1]
        assert tail_service >= config.NUM_MICROSERVICES, f"❌ 流 {i} 的尾端服务 {tail_service} 不是 AI 服务！"
    print("✅ 强制拓扑约束 (DAG 尾端强绑 AI 服务) 验证通过。")

    # ================= 2. 物理网络与张量路由基座装配 =================
    physical_net = PhysicalNetworkGraph()
    router = HeuristicRouter(physical_net)

    # 构造特定的实例分布以便于测试流量守恒
    N_matrix = np.zeros((config.NUM_NODES, config.NUM_SERVICES), dtype=np.int32)
    # 为避免全网无实例导致流量掉洞，我们为主机 1 和 2 强制各部署 1 个所有服务的实例
    N_matrix[1, :] = 1
    N_matrix[2, :] = 1

    # 执行微观时间尺度路由推演
    lambda_agg, F_tensor, P_tensor = router.step_route(N_matrix, generator)

    # ================= 3. Burke 定理流量守恒断言 =================
    print("\n--- 正在验证网络内部流量守恒定理 ---")
    for s in range(config.NUM_SERVICES):
        # 【修改这里】：忠于物理现实！计算该服务在每条链中出现的绝对频次，并叠加流量
        expected_total_lambda = sum(
            [f['lambda'] * np.sum(f['chain'] == s) for f in generator.active_flows]
        )
        
        # 计算物理侧 (Router) 实际在各个节点上该服务接收到的聚合请求总量
        actual_total_lambda = np.sum(lambda_agg[:, s])
        
        # 允许极微小的浮点数精度误差 (np.isclose)
        assert np.isclose(expected_total_lambda, actual_total_lambda, atol=1e-3), \
            f"❌ 服务 {s} 流量不守恒！预期总流入: {expected_total_lambda}, 实际总到达: {actual_total_lambda}"
    print("✅ 全网流量守恒 (Flow Conservation) 张量传递验证通过！无任何数据包掉洞丢失。")

    # ================= 4. 概率图谱归一化与非法节点过滤断言 =================
    # 节点 3 没有任何实例，断言 P 张量中发往节点 3 的概率绝对为 0
    assert np.all(P_tensor[:, :, 3] == 0.0), "❌ 智能路由器将流量发往了空壳节点！"
    
    # 仅针对那些全局确切拥有实例的服务，断言源节点的出向概率之和为 1
    for s in range(config.NUM_SERVICES):
        if np.sum(N_matrix[:, s]) > 0:
            prob_sum = np.sum(P_tensor[s], axis=1)
            # prob_sum 应该是一个全是 1.0 的一维向量 (长度为 V)
            assert np.allclose(prob_sum, 1.0), f"❌ 服务 {s} 的路由出向概率发生泄露，未严格归一化为 1.0！"
            
    print("✅ 自适应转发概率张量 (P Tensor) 完美归一化与物理拦截验证通过。")
    print("\n🎉 Day 3 里程碑达成！纯 Numpy 张量引擎实现了对毫秒级海量流量轨迹的高效数学刻画！")

if __name__ == "__main__":
    run_test()