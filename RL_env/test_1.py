"""
文件作用：Day 1 验收测试脚本。
目标：验证 Gym 环境输出的 Dict 观测空间是否具备对接 GNN 特征提取的合格张量形状与对齐属性。
"""
import numpy as np
from hybrid import HybridOrchestrationEnv
import config

def run_test():
    env = HybridOrchestrationEnv()
    obs, _ = env.reset()
    
    print(">>> 正在执行 Day 1 图拓扑与基座结构测试...\n")

    # 1. 张量对齐与字典存在性断言
    assert isinstance(obs, dict), "Observation 必须是字典类型"
    assert "macro_node_features" in obs, "缺失宏观节点特征"
    assert "macro_edge_features" in obs, "缺失宏观边特征"
    assert "micro_service_distribution" in obs, "缺失微观实例分布特征"
    print("✅ 状态空间 Dict 架构解耦验证通过。")

    # 2. 节点矩阵维度验证
    node_shape = (config.NUM_NODES, 3)
    assert obs["macro_node_features"].shape == node_shape, f"节点特征形状错误，应为 {node_shape}"
    assert np.all(obs["macro_node_features"] == 0.0), "环境初始状态下，资源利用率应严格为 0.0"
    print("✅ 节点特征矩阵 (Utilization Matrix) 验证通过。")

    # 3. 三维边矩阵通道验证
    edge_shape = (config.NUM_NODES, config.NUM_NODES, 2)
    assert obs["macro_edge_features"].shape == edge_shape, f"边特征形状错误，应为 {edge_shape}"
    # 测试同一节点内部带宽为无穷大，延迟为 0
    assert np.isinf(obs["macro_edge_features"][0, 0, 0]), "通道 0 (带宽) 主对角线应为 inf"
    assert obs["macro_edge_features"][0, 0, 1] == 0.0, "通道 1 (延迟) 主对角线应为 0.0"
    print("✅ GNN 三维边特征张量 (Edge Attributes) 组装验证通过。")

    print("\n🎉 Day 1 宏观图拓扑状态提取与封装里程碑已达成！现在底层基座已具备向图神经网络平滑演进的结构能力。")

if __name__ == "__main__":
    run_test()