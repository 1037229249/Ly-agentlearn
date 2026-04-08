"""
此模块用于管理仿真环境中的所有服务（微服务与AI服务）。
服务资源开销的描述，具象化为一个静态的 NumPy 矩阵 `service_req_matrix`。
它的维度是 (NUM_SERVICES, 3)，行代表不同的服务，列代表 (CPU, GPU, MEM) 的单实例开销。
利用这个矩阵，后续我们在环境中只需一行矩阵乘法，就能算出千万个实例的综合资源消耗。
"""
import numpy as np
import config

class ServiceRegistry:
    def __init__(self):
        # 初始化一个全零矩阵，维度为 (服务总数, 3) 对应 CPU, GPU, MEM
        self.service_req_matrix = np.zeros((config.NUM_SERVICES, 3), dtype=np.float32)
        # 调用内部方法生成服务配置
        self._generate_services()
    
    def _generate_services(self):
        # 遍历所有微服务索引 (0 到 NUM_MICROSERVICES - 1)
        for i in range(config.NUM_MICROSERVICES):
            # 微服务单实例 CPU 消耗为 1~2 Core
            cpu_req = np.random.uniform(1.0, 2.0)
            # 微服务不需要 GPU，设为 0
            gpu_req = 0.0
            #微服务单实例 MEM 消耗为 10~15 MB (统一转换为GB以便与节点单位对齐：0.01~0.015 GB)
            mem_req = np.random.uniform(10.0, 15.0) / 1024.0
            # 将微服务的资源需求向量写入矩阵的第 i 行
            self.service_req_matrix[i] = [cpu_req, gpu_req, mem_req]

        # 遍历所有 AI 服务索引 (接在微服务后面)
        for j in range(config.NUM_AI_SERVICES):
            # 计算在矩阵中的全局行号索引
            idx = config.NUM_MICROSERVICES + j

            #区分 Full-size AI (全量大模型) 与 Light-weight AI (轻量模型)
            if j == 0:# 假设第一个 AI 服务是 Full-size 强制云端模型
                # 全量模型消耗巨大，CPU 给定 16 核
                cpu_req = 16.0
                # 设定极高的 GPU 需求（如 4 张卡），确保其超出单一边缘节点上限(最大为2)，从而只能在云端合法部署
                gpu_req = 4.0
                # 内存需求给 32GB
                mem_req = 32.0
            else:# 其他 AI 服务为 Light-weight 模型，允许在边缘部署
                # 轻量模型 CPU 需求较低，设为 4 核
                cpu_req = 4.0
                # GPU 需求设为 1 张卡，符合边缘节点的 GPU 上限
                gpu_req = 1.0
                # 内存需求设为 4GB
                mem_req = 4.0

            # 将 AI 服务的资源需求向量写入矩阵的对应行
            self.service_req_matrix[idx] = [cpu_req, gpu_req, mem_req]