# 文件路径: Ly-agentlearn/tests/test_mini_train.py
"""
文件作用：基于现有代码库构建的“白盒测试与环境诊断工具”。
核心目标：
1. 采用单步颗粒度跟踪，通过控制台清晰的结构化日志输出，将环境内部的“动作语义”、“物理状态变化”及“因果惩罚”暴露出来。
2. 绝对不规避物理常识下的“联合溢出”（-100）惩罚问题，直观供人工核查与诊断。
3. 提供一套经过精算配平的环境参数，确保流量潮汐有解、学得到规律、且避免因跨回合截断破坏强化学习时序。

规范与配平说明 (Why this config works):
- Node & Service 比例: 1云 + 2边缘 = 3节点。搭载 2种微服务 + 1种 AI 服务 = 3服务。系统组合锐减为 9 维（3x3网格），这能使我们在终端极其直观地看清每一步的复合部署到底是增是减，不至于被庞大的矩阵淹没。
- Traffic Constraints: 原有的 1500 req/s 对于削减后的微型拓扑是毁灭性的。我们将 NUM_FLOWS 并发缩成 2 条，MAX_ARRIVAL_RATE 降到 100 req/s。该组合能在有限的 3 节点规模下保持“高负载但不至于立刻崩溃”的运行状态，使得资源利用率波动明显且策略有解。
- Episode Length & PPO Alignment: 为了充分观察潮汐变化并将错误行为暴露无遗，将 max_steps_per_episode 从原先的 128 拉长至 256 步。且重头戏在于：PPO 的 n_steps (单次收集轨迹长) 严格对齐 256 步，batch_size 定为 64 (无余量完美切分4批)。这保证了系统“绝对完整跑完一个回合后，再利用这整个回合的数据做一次网络更新”，避免了时序截断的逻辑撕裂。
"""

import sys
import os
import numpy as np

# ================= 1. 环境变量注入与模块路径配置 =================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RL_ENV_DIR = os.path.join(PROJECT_ROOT, 'RL_env')

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RL_ENV_DIR)

# ================= 2. 动态覆写 config 配置 (白盒探针的底层缩放) =================
# 在导入后续的环境逻辑模块前，必须率先拦截并重新定义底层常量。
import config

# [规模约束]
config.NUM_CLOUD_NODES = 2
config.NUM_EDGE_NODES = 10
config.NUM_NODES = config.NUM_CLOUD_NODES + config.NUM_EDGE_NODES

config.NUM_MICROSERVICES = 10
config.NUM_AI_SERVICES = 2
config.NUM_SERVICES = config.NUM_MICROSERVICES + config.NUM_AI_SERVICES

# [流量防决堤]
config.NUM_FLOWS = 5
config.MAX_ARRIVAL_RATE = 1500.0  

# ================= 3. 导入被测试的核心原生模块 =================
# 注意：受 Python 模块缓存机制影响，以下文件在底层 import config 时，读到的将是我们这套“微型诊断全家桶”数据
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from RL_env.hybrid import HybridOrchestrationEnv
from RL_env.environment_wrappers import DictObsNormalizationWrapper, MaskableWrapper


# ================= 4. 白盒诊断探针 (Diagnostic Callback) =================
class DiagnosticCallback(BaseCallback):
    def __init__(self, env_unwrapped, verbose=1):
        super(DiagnosticCallback, self).__init__(verbose)
        self.env_unwrapped = env_unwrapped
        
        # 内部状态流水累加器
        self.ep_steps = 0
        self.ep_reward = 0.0
        self.ep_invalid_count = 0

    def _on_step(self) -> bool:
        self.ep_steps += 1
        
        # sb3 内部截获网络执行后的瞬时数据 (单矢量环境直接提取 [0])
        action_flat = self.locals["actions"][0]
        reward = self.locals["rewards"][0]
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]
        
        # 提取动态设置的最大步数
        max_steps = self.env_unwrapped.max_steps_per_episode
        
        # 1. 打印宏观步骤头
        print(f"\n[Step {self.ep_steps}/{max_steps}]")
        
        # 2. Action 解码：将一维连续动作反转为物理增减矩阵 (0/1/2 重新平移为 -1/0/1)
        action_matrix = action_flat.reshape((config.NUM_NODES, config.NUM_SERVICES)) - 1
        action_strs = []
        for n in range(config.NUM_NODES):
            for s in range(config.NUM_SERVICES):
                delta = action_matrix[n, s]
                if delta != 0:
                    sign = "+1" if delta > 0 else "-1"
                    action_strs.append(f"N{n}S{s}({sign})")
                    
        action_desc = ", ".join(action_strs) if action_strs else "全系统静默 (Hold)"
        print(f"  > Action: {action_desc}")
        
        # 3. Reward/Penalty 因果诊断与拆解
        is_invalid = ("error_msg" in info and "Exceeds" in info.get("error_msg", ""))
        
        # 提取奖励分项 (如 Delay, Cost, Variance)，去掉 'Rewards/' 前缀以节省空间
        reward_details = {k.replace('Rewards/', ''): round(v, 2) for k, v in info.items() if k.startswith('Rewards/')}
        details_str = ", ".join([f"{k}:{v}" for k, v in reward_details.items()]) if reward_details else "无细分"
        
        if is_invalid:
            self.ep_invalid_count += 1
            print(f"  > Reward: {reward:.1f} | 🔴 溢出惩罚(-100！多动作联合突破物理极限)")
        else:
            print(f"  > Reward: {reward:.2f} | 🟢 决算明细: [{details_str}]")
            
        # 4. State Highlight 摘要提取：穿透 Wrapper 直接深入物理引擎抽水
        try:
            # 流量潮汐捕捉：由于泊松过程，此处能看到非常清楚的数值摇摆
            flows = self.env_unwrapped.traffic_gen.active_flows
            flow_rates = [f"F{f['flow_id']}:{f['lambda']:.1f}q/s" for f in flows]
            
            # 算力负载捕捉：将第一维(CPU Util)视作承压特征代表
            util_matrix = self.env_unwrapped.physical_net.get_utilization_matrix()
            cpu_utils = [f"N{i}:{util_matrix[i][0]:.1%}" for i in range(config.NUM_NODES)]
            
            state_desc = f"内部排队流量 [{', '.join(flow_rates)}] | 节点CPU利用率 [{', '.join(cpu_utils)}]"
        except Exception as e:
            state_desc = f"强行解包底层状态时发生异常: {str(e)}"
            
        print(f"  > State Highlight: {state_desc}")
        
        # 记录累积奖励
        self.ep_reward += reward
        
        # 5. Episode 清算与小结
        # 注意：SB3中 truncated 同样包含在 dones 里
        if done or self.ep_steps >= max_steps:
            print("\n=======================================================")
            print(f"🔄 Episode Summary (回合结算小结)")
            print(f"  - 总存活时长: {self.ep_steps} Steps")
            print(f"  - 整体策略奖励 (Total Reward): {self.ep_reward:.2f}")
            print(f"  - 资源预判不足撞墙惩罚次数: {self.ep_invalid_count} 次")
            print("=======================================================")
            
            # 清理重置
            self.ep_steps = 0
            self.ep_reward = 0.0
            self.ep_invalid_count = 0
            
        return True


def main():
    print("=========================================================================================")
    print("🚀 启动自动化 PPO “白盒测试与环境诊断探针” (Test Mini Train)")
    print("目标定位：以微缩自洽模型暴露底层设计漏洞，重在人工观察“动作->环境受击->硬约束截留”的因果时序。")
    print("=========================================================================================\n")
    
    # 1. 组建物理环境底座
    raw_env = HybridOrchestrationEnv()
    
    # 核心配平：强制从外部覆盖拉升回合长度，确保能够跨越整段潮汐时间
    TEST_MAX_STEPS = 256
    raw_env.max_steps_per_episode = TEST_MAX_STEPS
    
    # 装载洋葱 Wrapper
    env = DictObsNormalizationWrapper(raw_env)
    env = MaskableWrapper(env)
    
    # 填入自研白盒探针（保留底座 raw_env 的纯裸引用，以越过 Normalize 做深层透视）
    diagnostic_callback = DiagnosticCallback(env_unwrapped=raw_env)
    
    # 2. 组装 MaskPPO 更新系统
    # [参数重度绑定解释]
    # n_steps 等于最新最大步数 256，这意味着模型只在“此完整回合彻底结束时”才闭环一次学习更新。
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=TEST_MAX_STEPS,
        batch_size=64,           # 完美被 256 取模，批次计算绝不含余数
        gamma=0.99,
        verbose=0                # 彻底屏蔽 SB3 自带的日志流，只看咱们的高亮日志
    )
    
    # 诊断目的：仅预演 2 个完整对齐的巨型回合。合计 512 步
    TOTAL_EXEC_STEPS = TEST_MAX_STEPS * 2
    
    try:
        model.learn(
            total_timesteps=TOTAL_EXEC_STEPS,
            callback=diagnostic_callback,
            progress_bar=False       # 防止进度条反复吞噬控制台日志
        )
        print("\n✅ 诊断探测脚本运行平稳结束！请往上滚屏查阅排查日志。")
    except Exception as e:
        import traceback
        print("\n❌ 测试过程发生意外断裂！诊断现场信息：")
        traceback.print_exc()

if __name__ == "__main__":
    main()
