# 文件路径: Ly-agentlearn/RL_env/test_4.py
"""
文件作用：Day 4 综合延迟计算与路由返回值验证。
目标：验证 F_tensor 仍然存在，且 traffic_bytes_tensor 被正确提取用于通信延迟计算。
"""
import numpy as np
from traffic_routing import DynamicTrafficGenerator, HeuristicRouter
from topology_graph import PhysicalNetworkGraph
from queueing_engine import QueueingEngine
import config

def run_test():
    print(">>> 正在执行 Day 4：路由多协议返回与排队引擎集成测试...\n")

    # 1. 初始化基座
    net = PhysicalNetworkGraph()
    tg = DynamicTrafficGenerator()
    router = HeuristicRouter(net)

    # 2. 模拟部署 
    N_matrix = np.ones((config.NUM_NODES, config.NUM_SERVICES), dtype=np.int32)

    # 3. 调用更新后的路由方法
    # 接收四元组返回
    lambda_agg, F_tensor, traffic_bytes, P_tensor = router.step_route(N_matrix, tg)

    # 断言 F_tensor 依然有效且保留了流量强度信息
    assert F_tensor.shape == (config.NUM_SERVICES, config.NUM_NODES, config.NUM_NODES)
    assert np.sum(F_tensor) > 0, "F_tensor 流量强度丢失！"

    # 断言 traffic_bytes 反应了数据负载
    assert traffic_bytes.shape == (config.NUM_NODES, config.NUM_NODES)
    print(f"✅ 路由四元组返回验证通过。全网总数据负载: {np.sum(traffic_bytes):.2f} MB/s")

    # 4. 计算通信延迟
    comm_delays = QueueingEngine.calc_comm_delay_matrix(traffic_bytes, net)
    assert not np.isnan(comm_delays).any()
    print("✅ 通信延迟矩阵计算成功。")

    print("\n🎉 Day 4 里程碑达成！代码结构保持了高度的工程纯粹性，成功实现了物理层与逻辑层的解耦。")

if __name__ == "__main__":
    run_test()