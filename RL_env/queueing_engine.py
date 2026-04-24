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
from scipy.special import gammaln, pdtr
import config

class QueueingEngine:
    @staticmethod
    #@staticmethod 装饰器表示该方法是一个静态方法，可以直接通过类名调用，而不需要实例化对象。
    def calc_mmc_delay_tensor(Lambda, Mu, C):
        """
        计算微服务 (M/M/c) 延迟矩阵。
        输入维度：Lambda, Mu, C 均为 (NUM_NODES, NUM_MICROSERVICES)
        """
        C_float = C.astype(np.float32)  # 将实例数矩阵转换为浮点数类型，便于后续计算

       # 1. 物理常识断言
        if np.any(Mu <= 0):
            raise ValueError("异构算力异常：服务处理速率 Mu 必须严格大于 0！")
        if np.any(Lambda < 0):
            raise ValueError("流量异常：到达率 Lambda 不能为负数！")
        
        # 2. 物理三态解析
        # 状态 A：绝对静止（没有流量，延迟必定为 0.0）
        mask_zero_traffic = (Lambda == 0.0)

        # 状态 B：系统宕机与雪崩（有流量但没实例，或流量击穿算力上限）
        # 安全张量除法：A = Lambda / Mu
        A = np.divide(Lambda, Mu, out=np.zeros_like(Lambda), where=(Mu > 0))
        # 安全计算利用率 Rho
        Rho = np.divide(A, C_float, out=np.zeros_like(A), where=(C_float > 0))
        mask_overload = (Lambda > 0.0) & ((C == 0) | (Rho >= 1.0))

        # 状态 C：合法排队稳态
        mask_valid = ~(mask_zero_traffic | mask_overload)

        # 3. 稳态运算安全区隔离 (填入 dummy variables 防止 numpy 底层 math domain error)
        A_valid = np.where(mask_valid, A, 1.0)
        C_valid = np.where(mask_valid, C_float, 1.0)
        Rho_valid = np.where(mask_valid, Rho, 0.5)

        # 对数空间 Erlang C 概率推导 (消除 c! 溢出)
        cdf = poisson.cdf(C_valid - 1, A_valid)  # 计算泊松分布的累积分布函数，参数为 C_valid - 1 和 A_valid
        cdf_safe = np.where(cdf == 0, 1e-30, cdf)  # 防止累积概率为零导致的数值问题

        log_term = A_valid + np.log(cdf_safe) + gammaln(C_valid + 1) - C_valid * np.log(A_valid)
        term = np.exp(log_term)  # 将对数空间的计算结果转换回正常空间

        # 计算多服务台繁忙概率Pc
        Pc = 1.0 / (term * (1.0 - Rho_valid) + 1.0)

       # 稳态 Erlang C 期望延迟
        denominator = np.maximum(C_valid * Mu - Lambda, 1e-5) # 防止除零
        expected_delay = Pc / denominator + 1.0 / Mu

        # 4. 物理最终结果组装
        final_delay = np.zeros_like(Lambda, dtype=np.float32)
        final_delay[mask_overload] = config.MAX_DELAY_MS
        final_delay[mask_valid] = expected_delay[mask_valid]

        return final_delay

    @staticmethod
    def calc_mm1_delay_tensor(Lambda, Mu, C):
        """
        计算 AI 服务 (M/M/1) 批处理延迟矩阵。
        输入维度：Lambda, Mu, C 均为 (NUM_NODES, NUM_AI_SERVICES)
        """
        C_float = C.astype(np.float32)  # 将实例数矩阵转换为浮点数类型，便于后续计算

        if np.any(Mu <= 0):
            raise ValueError("异构算力异常：服务处理速率 Mu 必须严格大于 0！")

        # 物理三态解析
        mask_zero_traffic = (Lambda == 0.0)

        # 每个实例分摊的到达率
        lambda_inst = np.divide(Lambda, C_float, out=np.zeros_like(Lambda), where=(C_float > 0))
        mask_overload = (Lambda > 0.0) & ((C == 0) | (lambda_inst >= Mu))
        mask_valid = ~(mask_zero_traffic | mask_overload)

        # 安全计算区
        lambda_inst_valid = np.where(mask_valid, lambda_inst, 0.0)
        Mu_valid = np.where(mask_valid, Mu, 1.0)
        
        # M/M/1 期望延迟计算公式
        expected_delay = 1.0 / (Mu_valid - lambda_inst_valid)

        # 物理组装
        final_delay = np.zeros_like(Lambda, dtype=np.float32)
        final_delay[mask_overload] = config.MAX_DELAY_MS
        final_delay[mask_valid] = expected_delay[mask_valid]

        return final_delay
        
    
    @staticmethod
    # 参数说明：traffic_bytes_tensor 是一个形状为 (NUM_NODES, NUM_NODES) 的张量，表示每对节点之间的流量大小（以字节为单位）。
    # physical_net 是一个包含网络拓扑信息的对象，能够提供节点之间的物理距离或带宽等信息。
    def calc_comm_delay_matrix(traffic_bytes_tensor, physical_net):
        """计算全网链路的通信延迟矩阵 (NUM_NODES, NUM_NODES)"""
        B_matrix = physical_net.edge_features[:,:,0]  # 提取带宽矩阵 B_{i,j}
        D_matrix = physical_net.edge_features[:,:,1]  # 提取物理延迟矩阵 d_{i,j}

        # 简化计算：MB * 8 / Mbps = 秒 -> 换算为 ms
        # 实际工程中需注意单位：1 MB = 8 Mbits
        # 1. 物理越界断言 (带宽断开但流量堆积)
        if np.any((B_matrix == 0) & (traffic_bytes_tensor > 0)):
            raise ValueError("物理路由越界：尝试在带宽为 0 的断开链路上进行数据传输！")

        # 安全除法：零流量时传输延迟严格为 0
        transmission_delay = np.divide(
            traffic_bytes_tensor * 8.0, 
            B_matrix, 
            out=np.zeros_like(traffic_bytes_tensor), 
            where=(B_matrix > 0)
        )

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