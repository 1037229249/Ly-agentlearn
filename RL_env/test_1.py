"""
1 验收脚本。用于验证环境实例的初始化是否成功，输出的数据结构、维度以及取值
是否完全契合指南中的数学定义与代码规范。
"""
import numpy as np
from hybrid import HybridOrchestrationEnv
import config

def run_test():
    # 创建环境实例
    env = HybridOrchestrationEnv()

    # 重置环境，获取初始状态
    obs, info = env.reset()
   #assert是Python中的一个断言语句，用于测试一个条件是否为真。如果条件为假，assert语句会抛出一个AssertionError异常，并显示指定的错误消息。
   # 2. 验证返回数据类型
   #assert isinstance作用是检查obs是否是一个字典类型。
    assert isinstance(obs, dict), "Observation 必须是字典类型"
    assert isinstance(info, dict), "Info 必须是字典类型"
    print("✅ 数据类型断言通过")
    
    # 3. 验证矩阵维度
    expected_shape = (config.NUM_NODES, 3)
    #assert obs[].shape == expected_shape中的obs["node_remaining_resources"]作用是从obs字典中获取键为"node_remaining_resources"的值
    # 这个值应该是一个矩阵（二维数组）。然后通过.shape属性获取这个矩阵的维度，并与预期的维度expected_shape进行比较。
    assert obs["node_remaining_resources"].shape == expected_shape, f"剩余资源矩阵维度错误，应为 {expected_shape}"
    assert obs["node_resource_utilization"].shape == expected_shape, f"利用率矩阵维度错误，应为 {expected_shape}"
    print("✅ 矩阵维度断言通过")
    
    # 4. 验证异构性逻辑（云节点拥有最大资源）
    cloud_cpu = obs["node_remaining_resources"][0, 0] # 节点0 的 CPU 维度
    edge_1_cpu = obs["node_remaining_resources"][1, 0] # 节点1 的 CPU 维度
    assert cloud_cpu > edge_1_cpu * 10, "云端算力必须显著大于单一边缘节点算力"
    print("✅ 异构性设定验证通过")
    
    # 5. 验证初始状态的边界值
    assert np.all(obs["node_resource_utilization"] == 0.0), "初始状态下资源利用率必须全为 0.0"
    print("✅ 初始利用率验证通过")
    
    print("🎉 恭喜！Day 1 的底层基座开发完全符合规范，基础非常牢靠！")

if __name__ == "__main__":
    run_test()