"""
文件作用：排队论计算引擎。
功能：实现 M/M/c 与 M/M/1 混合排队网络的纯张量并行计算。
工程亮点：使用 scipy.stats.poisson.cdf 与 scipy.special.gammaln 将 Erlang C 公式
转换至对数空间（Log-space），彻底解决大并发实例数下的 float64 阶乘数值溢出问题。
"""
import numpy as np
# 从 scipy 库中导入泊松分布的累积分布函数 (CDF)，用于计算请求到达率的概率分布
from scipy.stats import poisson
# 从 scipy 库中导入 gammaln 函数，用于计算阶乘的对数值，避免大数阶乘导致的数值溢出
from scipy.special import gammaln
import config

class QueueingEngine:
    @staticmethod
    #@staticmethod 装饰器表示该方法是一个静态方法，可以直接通过类名调用，而不需要实例化对象。
    def calc_mmc_delay_tensor(Lamda, Mu, C):
        """
        计算微服务 (M/M/c) 延迟矩阵。
        输入维度：Lambda, Mu, C 均为 (NUM_NODES, NUM_MICROSERVICES)
        """
        C_float = C.astype(np.float32)  # 将实例数矩阵转换为浮点数类型，便于后续计算

        # 避免除零异常
        Mu_safe = np.where(Mu == 0, 1e-9, Mu)  # 将处理速率为零的元素替换为一个非常小的数，防止除零错误
        C_safe = np.where(C == 0, 1e-9, C_float) 

        # Rho代表服务强度，即请求到达率与处理能力的比值，计算公式为 Rho = (Lambda / Mu) / C
        A = Lamda / Mu_safe  
        Rho = A / C_safe  

        # 识别超载或者无实例的非法区域
        mask_overload = (Rho >= 1.0) | (C == 0)  # 当服务强度大于等于1或实例数为0时，系统处于过载状态
        # 隔离合法数据区域，防止非法数据在scipy函数中引发数值错误
        A_valid = np.where(mask_overload, 1e-9, A)
        C_valid = np.where(mask_overload, 1, C_float)
        Rho_valid = np.where(mask_overload, 0.5, Rho)

        # 对数空间 Erlang C 概率推导 (消除 c! 溢出)
        cdf = poisson.cdf(C_valid - 1, A_valid)  # 计算泊松分布的累积分布函数，参数为 C_valid - 1 和 A_valid
        cdf_safe = np.where(cdf == 0, 1e-30, cdf)  # 防止累积概率为零导致的数值问题

        log_term = A_valid + np.log(cdf_safe) + gammaln(C_valid + 1) - C_valid * np.log(A_valid + 1e-9)
        term = np.exp(log_term)  # 将对数空间的计算结果转换回正常空间

        # 计算多服务台繁忙概率Pc
        Pc = 1.0 / (term * (1.0 - Rho_valid) + 1.0)

        # Erlang C 期望延迟
        delay = Pc / (C_valid * Mu_safe - Lamda) + 1.0 / Mu_safe

        # 将非法区域的延迟设置为一个非常大的数，表示系统崩溃
        return np.where(mask_overload, config.MAX_DELAY_MS, delay)

    @staticmethod
    def calc_mm1_delay_tensor(Lamda, Mu, C):
        """
        计算 AI 服务 (M/M/1) 批处理延迟矩阵。
        输入维度：Lambda, Mu, C 均为 (NUM_NODES, NUM_AI_SERVICES)
        """
        C_float = C.astype(np.float32)  # 将实例数矩阵转换为浮点数类型，便于后续计算
        # 避免除零异常
        C_safe = np.where(C == 0, 1e-9, C_float)

        # 请求均匀分布到每个实例上，计算每个实例的请求到达率
        lambda_inst = Lamda / C_safe  # 每个实例的请求到达率

        # 稳定性约束
        mask_overload = (lambda_inst >= Mu) | (C == 0)  # 当每个实例的请求到达率大于等于处理速率或实例数为0时，系统处于过载状态

        Mu_safe = np.where(mask_overload, 1, Mu)
        lambda_inst_safe = np.where(mask_overload, 0.0, lambda_inst) 

        # M/M/1 期望延迟计算公式：D = 1 / (Mu - lambda_inst)
        delay = 1.0 / (Mu_safe - lambda_inst_safe)
        # where 作用：当 mask_overload 条件为 True 时，返回 config.MAX_DELAY_MS；否则返回计算得到的 delay 值。这确保了在系统过载或无实例的情况下，延迟被设置为一个非常大的数，表示系统崩溃。
        return np.where(mask_overload, config.MAX_DELAY_MS, delay)  
    
    @staticmethod
    # 参数说明：traffic_bytes_tensor 是一个形状为 (NUM_NODES, NUM_NODES) 的张量，表示每对节点之间的流量大小（以字节为单位）。
    # physical_net 是一个包含网络拓扑信息的对象，能够提供节点之间的物理距离或带宽等信息。
    def calc_comm_delay_matrix(traffic_bytes_tensor, physical_net):
        """计算全网链路的通信延迟矩阵 (NUM_NODES, NUM_NODES)"""
        B_matrix = physical_net.edge_features[:,:,0]  # 提取带宽矩阵 B_{i,j}
        D_matrix = physical_net.edge_features[:,:,1]  # 提取物理延迟矩阵 d_{i,j}

        # 简化计算：MB * 8 / Mbps = 秒 -> 换算为 ms
        # 实际工程中需注意单位：1 MB = 8 Mbits
        transmission_delay = (traffic_bytes_tensor * 8.0) / np.where(B_matrix == 0, 1e-9, B_matrix)
        return transmission_delay + D_matrix

    @staticmethod
    def calculate_end_to_end_delay(traffic_generator, P_tensor, node_delays, comm_delay_matrix) :
        """基于路由概率矩阵与有向无环图，计算每条流的端到端加权期望延迟"""
        expected_delays = []

        for flow in traffic_generator.active_flows:
            chain = flow['chain']
            start_node = flow['start_node']

            # 使用概率分布向量在图谱中游走
            current_prob = np.zeros(config.NUM_NODES, dtype=np.float32)
            current_prob[start_node] = 1.0  # 从起始节点开始，概率为 1

            flow_delay = 0.0
            pre_s = None

            for s in chain:
                # 累加节点（处理+排队）延迟期望
                flow_delay += np.sum(current_prob * node_delays[:, s])

                # 累加跨节点通信延迟期望
                if pre_s is not None:
                    step_comm_delay = np.sum(
                        current_prob[:, np.newaxis] * P_tensor[pre_s] * comm_delay_matrix
                                             )
                    flow_delay += step_comm_delay

                # 马尔可夫状态转移至下一跳
                current_prob = current_prob @ P_tensor[s]
                pre_s = s
            
            expected_delays.append(flow_delay)

        return np.array(expected_delays, dtype=np.float32)