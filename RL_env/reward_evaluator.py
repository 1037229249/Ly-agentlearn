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

        # ------------------ CMDP RCPO 理论核心参数 (精准对齐 PPO LR) ------------------
        self.dynamic_lambda_sla = 0.0  
        
        # 【关键数学推理】：
        # PPO 的 Actor 学习率是 3e-4。在 CMDP 中，拉格朗日乘子 λ 相当于“环境在对抗智能体”。
        # 根据 Two-Timescale 定理，λ 的更新速率（外环）必须严格小于 Actor 的学习率（内环），
        # 否则环境变化比智能体学习还快，模型永远无法收敛。
        # 因此，alpha_lambda 设定为 3e-6（比 3e-4 慢一个数量级）。
        self.alpha_lambda = 3e-6     
        
        # 遗忘因子。稳定状态下，λ 的理论上限 = alpha / leak * 最大梯度
        # 最大梯度约等于 1.0 (0.99 - 0.0)。
        # 如果 leak = 3e-6，那么 λ_max = 3e-6 / 3e-6 * 1.0 = 1.0。
        # 这完美匹配了 r_base (最大为 1.0) 的量级，绝对不会发生 λ 飙升导致奖励崩溃。
        self.lambda_leak_rate = 3e-6  
        self.max_lambda = 1.5      # 物理绝对上限

        # 宏观动作仅为增减1，单步变动上限是节点数 × 服务数
        self.max_ai_delta_per_step = config.NUM_NODES * config.NUM_AI_SERVICES
        self.max_truncated_per_step = config.NUM_NODES * config.NUM_SERVICES

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
    
    def evaluate_step_reward(self, total_delay_array, utilization_matrix, N_matrix_current, N_matrix_prev, truncated_action_count):
        """
        核心多目标奖励计算函数
        计算当前时间步的综合奖励。
        total_delay_array: (NUM_FLOWS,) 每条调用链的总延迟数组。
        utilization_matrix: (NUM_NODES, 3) 当前节点资源利用率矩阵。
        N_matrix_current: (NUM_NODES, NUM_SERVICES) 当前时间步的服务实例分布矩阵。
        N_matrix_prev: (NUM_NODES, NUM_SERVICES) 上一个时间步的服务实例分布矩阵。
        返回一个标量 reward 和一个 info 字典，包含详细的物理指标供 Tensorboard 记录。
        """
        if len(total_delay_array) == 0:
            raise ValueError("逻辑异常：当前时间步没有任何活跃的请求流（流量矩阵为空），无法评估 SLA！")
        # ================= 1. 物理指标统计 =================
        # np.mean表示计算数组的平均值，total_delay_array是一个包含每条调用链总延迟的数组，avg_delay是所有调用链总延迟的平均值。
        avg_delay = np.mean(total_delay_array)  # 平均延迟

        # SLA 成功率: 延迟小于容忍上限的流的比例
        success_count = np.sum(total_delay_array <= config.QOS_DELAY_TOLERANCE)
        current_success_rate = success_count / len(total_delay_array)

        v_load = self.compute_load_variance(utilization_matrix)  # 负载方差
        c_total = self.compute_deployment_cost(N_matrix_current)  # 部署成本

        # ================= 1. 上层核心任务奖励 (Task Rewards)：平滑指数映射 =================
        # np.exp(-x/scale) 保证了结果在 (0, 1] 且全程连续可导
        score_delay = np.exp(-avg_delay / config.SCALE_DELAY_MS)
        score_var = np.exp(-v_load / config.SCALE_VAR)
        score_cost = np.exp(-c_total / config.SCALE_COST)

        r_base = (config.ETA_DELAY * score_delay + 
                  config.ETA_VARIANCE * score_var + 
                  config.ETA_COST * score_cost)
        
        # ================= 3. CMDP 理论 (移除 Clip，依靠自然收敛) =================
        sla_gradient = config.TARGET_SLA_SUCCESS - current_success_rate
        self.dynamic_lambda_sla = (1.0 - self.lambda_leak_rate) * self.dynamic_lambda_sla + self.alpha_lambda * sla_gradient
        sla_violation_penalty_term = max(0.0, sla_gradient) 
        penalty_sla = - self.dynamic_lambda_sla * sla_violation_penalty_term

        # ================= 4. 线性密集引导惩罚 (彻底摒弃对数，保持恒定痛感梯度) =================
        # 根据推导：截断每次扣 0.05，抖动每次扣 0.05
        # 这将使随机探索期的智能体每步吃到 -1.0 左右的稳定负反馈
        penalty_intent = - 0.05 * truncated_action_count

        ai_idx_start = config.NUM_MICROSERVICES
        delta_ai_instances = np.sum(np.abs(N_matrix_current[:, ai_idx_start:] - N_matrix_prev[:, ai_idx_start:]))
        penalty_smoothness = - 0.05 * delta_ai_instances

        # ================= 5. 性能驱动的动态门控 (Curriculum Gating) =================
        # 当截断+抖动总数 > 15 时，W_task 趋近于 0。
        # 迫使智能体前期 100% 精力用于规避违规，后期自动无缝过渡到优化 r_base。
        compliance_ratio = np.exp(-(truncated_action_count + delta_ai_instances) / 10.0)
        
        W_task = compliance_ratio

        # 最终奖励计算
        final_reward = W_task * (r_base + penalty_sla) + (penalty_intent + penalty_smoothness)

        info = {
            "Metrics/Avg_EndToEnd_Delay_ms": avg_delay,
            "Metrics/SLA_Success_Rate": current_success_rate,
            "Metrics/Load_Variance": v_load,
            "Metrics/Total_Cost": c_total,
            "Metrics/AI_Instance_Thrashing": delta_ai_instances,
            "Metrics/Truncated_Actions": truncated_action_count, # 监控模型自身“守规矩”的程度
            "Rewards/R_base": r_base,
            "Rewards/Penalty_SLA": penalty_sla,
            "Rewards/Penalty_Smoothness": penalty_smoothness,
            "Rewards/Penalty_Intent": penalty_intent,
            "CMDP/Dynamic_Lambda_SLA": self.dynamic_lambda_sla  # 输出到日志以便你在 CSV 中观察它的震荡情况
        }

        return float(final_reward), info