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

        