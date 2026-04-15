"""
文件作用：多目标奖励评估与物理指标提取器 (Reward Evaluator)。
包含模块：
1. 提取物理层的负载方差、部署成本、SLA 违规率。
2. 将多维度指标归一化，基于拉格朗日松弛法与 Reward Shaping 计算最终标量 Reward。
3. 生成详尽的 Academic Info 字典，供 Tensorboard 回调函数抓取绘制科研图表。
"""
import numpy as np
import config

class RewardEvaluator:
    def __init__(self):
        # 预构建成本单价向量 (长度为 NUM_SERVICES)
        self.cost_vector = np.zeros(config.NUM_SERVICES, dtype=np.float32)
        self.cost_vector[:config.NUM_MICROSERVICES] = config.COST_MICROSERVICE
        # 第一种AI为Full-size，其余为Light-weight
        self.cost_vector[config.NUM_MICROSERVICES] = config.COST_AI_FULL
        self.cost_vector[config.NUM_MICROSERVICES + 1:] = config.COST_AI_LIGHT

    def compute_load_variance(self, utilization_matrix):
        """
        计算全网各节点资源利用率的方差。
        utilization_matrix: (NUM_NODES, 3) 的矩阵，包含 CPU、GPU、内存的利用率。
        返回一个标量，表示三个资源维度的平均利用率方差。
        """
        # utilization_matrix 形状 (NUM_NODES, 3) 对应 CPU, GPU, MEM
        # 仅计算边缘节点 (索引 1 开始) 的方差，云端节点不计入负载均衡考核
        edge_utilization = utilization_matrix[1:]  # 形状 (NUM_NODES-1, 3)
        var_cpu = np.var(edge_utilization[:, 0])  # CPU 利用率方差
        var_gpu = np.var(edge_utilization[:, 1])  # GPU 利用率方差
        var_mem = np.var(edge_utilization[:, 2])  # 内存利用率方差
        # 简单赋予三维资源同等权重
        return (var_cpu + var_gpu + var_mem) / 3.0
    
    def compute_deployment_cost(self, N_matrix):
        """计算当前时间步的服务部署与激活成本"""
        # 实例部署成本: 实例数 * 对应单价
        # N_matrix 形状 (NUM_NODES, NUM_SERVICES)，每个元素表示该节点上该服务的实例数
        deploy_cost = np.sum(N_matrix * self.cost_vector)  # 广播乘法后求和得到总成本

        # 节点激活成本（如果边缘节点上部署了 > 0 个实例，则算作激活）
        # 统计边缘节点
        edge_active_mask = np.sum(N_matrix[1:], axis=1) > 0  # 形状 (NUM_NODES-1,) 的布尔数组，表示哪些边缘节点被激活
        active_cost = np.sum(edge_active_mask) * config.COST_NODE_ACTIVE

        return deploy_cost + active_cost
    
    def evaluate_step_reward(self, total_delay_array, utilization_matrix, N_matrix_current, N_matrix_prev):
        """
        核心多目标奖励计算函数
        计算当前时间步的综合奖励。
        total_delay_array: (NUM_FLOWS,) 每条调用链的总延迟数组。
        utilization_matrix: (NUM_NODES, 3) 当前节点资源利用率矩阵。
        N_matrix_current: (NUM_NODES, NUM_SERVICES) 当前时间步的服务实例分布矩阵。
        N_matrix_prev: (NUM_NODES, NUM_SERVICES) 上一个时间步的服务实例分布矩阵。
        返回一个标量 reward 和一个 info 字典，包含详细的物理指标供 Tensorboard 记录。
        """
        # ================= 1. 物理指标统计 =================
        # np.mean表示计算数组的平均值，total_delay_array是一个包含每条调用链总延迟的数组，avg_delay是所有调用链总延迟的平均值。
        avg_delay = np.mean(total_delay_array)  # 平均延迟

        # SLA 成功率: 延迟小于容忍上限的流的比例
        success_count = np.sum(total_delay_array <= config.QOS_DELAY_TOLERANCE)
        current_success_rate = success_count / (len(total_delay_array) + 1e-9)  # 避免除零错误

        v_load = self.compute_load_variance(utilization_matrix)  # 负载方差
        c_total = self.compute_deployment_cost(N_matrix_current)  # 部署成本

        # ================= 2. 归一化处理  =================
        norm_delay = np.clip(avg_delay / config.MAX_NORM_DELAY, 0.0, 1.0)
        norm_var = np.clip(v_load / config.MAX_NORM_VAR, 0.0, 1.0)
        norm_cost = np.clip(c_total / config.MAX_NORM_COST, 0.0, 1.0)

        # ================= 3. 基础联合多目标奖励 (均为惩罚，故取负) =================
        r_base = - (config.ETA_DELAY * norm_delay + 
                    config.ETA_VARIANCE * norm_var + 
                    config.ETA_COST * norm_cost)
        
        # ================= 4. 拉格朗日软约束 (SLA 惩罚) =================
        # 如果成功率不达标，给予动态惩罚
        sla_violation = max(0.0, config.TARGET_SLA_SUCCESS - current_success_rate)
        penalty_sla = - config.LAMBDA_SLA_PENALTY * sla_violation

        # ================= 5. Reward Shaping: AI 容器抖动惩罚 =================
        # 仅针对 AI 服务计算实例数量的绝对差值 (避免频繁启停)
        # 这里[:, ai_idx_start:] 是为了只计算 AI 服务的实例变动，微服务的变动不计入抖动惩罚。
        ai_idx_start = config.NUM_MICROSERVICES
        delta_ai_instances = np.sum(np.abs(
            N_matrix_current[:, ai_idx_start:] - N_matrix_prev[:, ai_idx_start:]
            ))
        penalty_smoothness = - config.OMEGA_SMOOTHNESS * delta_ai_instances

        # =================  Reward 与 Info 字典 =================
        final_reward = r_base + penalty_sla + penalty_smoothness

        # 供 Tensorboard 记录
        info = {
            "Metrics/Avg_EndToEnd_Delay_ms": avg_delay,
            "Metrics/SLA_Success_Rate": current_success_rate,
            "Metrics/Load_Variance": v_load,
            "Metrics/Total_Cost": c_total,
            "Metrics/AI_Instance_Thrashing": delta_ai_instances,
            "Rewards/R_base": r_base,
            "Rewards/Penalty_SLA": penalty_sla,
            "Rewards/Penalty_Smoothness": penalty_smoothness
        }

        return float(final_reward), info