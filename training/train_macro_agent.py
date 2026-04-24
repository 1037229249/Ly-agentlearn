"""
文件作用：宏观智能体 (Deployer) 全量强化学习训练主线。
工程规范：本文件完全静默运行，所有物理状态由 Callback 代理记录。
采用 sb3_contrib 的 MaskablePPO 算法，原生支持非法动作屏蔽。
"""
import sys
import os
import datetime
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from typing import Callable
from stable_baselines3.common.callbacks import BaseCallback

class EntCoefDecayCallback(BaseCallback):
    """
    用于在训练过程中动态调整熵系数 (ent_coef) 的回调函数。
    """
    def __init__(self, initial_ent_coef: float, final_ent_coef: float, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # 计算进度 (从 1.0 降到 0.0)
        progress = 1.0 - (self.num_timesteps / self.total_timesteps)
        progress = max(0, progress)
        
        # 计算当前的熵系数
        current_ent_coef = progress * (self.initial_ent_coef - self.final_ent_coef) + self.final_ent_coef
        
        # 注入到模型中
        self.model.ent_coef = current_ent_coef
        return True
# ================= 动态调度器生成函数 =================
def linear_schedule(initial_value: float, final_value: float = 0.0) -> Callable[[float], float]:
    """
    随着训练进度 (progress_remaining 从 1.0 减小到 0.0)
    平滑地将值从 initial_value 降到 final_value。
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func

# ================= 环境变量注入 =================
# 1. 获取当前脚本所在目录的上一级目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 2. 获取 RL_env 文件夹的具体路径
RL_ENV_DIR = os.path.join(PROJECT_ROOT, 'RL_env')

# 3. 将这两个路径插入到 Python 系统搜索路径的最前面
# 这样不仅能使用 from RL_env.xxx import yyy，还能让底层代码直接 import config
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RL_ENV_DIR)
# ==========================================================

from sb3_contrib import MaskablePPO
from RL_env.hybrid import HybridOrchestrationEnv
from RL_env.environment_wrappers import DictObsNormalizationWrapper, MaskableWrapper
from RL_env.callbacks import AcademicMetricsCallback

# 定义存储路径
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')

# 1. 获取当前时间并格式化为字符串
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 2. 将时间戳嵌入文件名中
CSV_PATH = os.path.join(LOG_DIR, f'ppo_macro_metrics_{timestamp}.csv')

# 3. 模型建议也加上时间戳，防止覆盖
MODEL_PATH = os.path.join(LOG_DIR, f'ppo_macro_agent_{timestamp}')

def main():
    print("🚀 正在启动云边协同混合编排智能体 (Macro-Deployer) 训练...")
    
    # 1. 组装环境洋葱模型
    env = HybridOrchestrationEnv()
    env = DictObsNormalizationWrapper(env)
    env = MaskableWrapper(env)
    

    TOTAL_TIMESTEPS = 300000

    # 2. 组装学术日志记录器
    academic_callback = AcademicMetricsCallback(
        save_path=CSV_PATH, 
        save_freq_episodes=50, # 每跑50个回合存一次盘
        verbose=1              # 开启控制台简报
    )
    
    # 熵系数衰减回调
    ent_decay_callback = EntCoefDecayCallback(
        initial_ent_coef=0.01, 
        final_ent_coef=0.0001, 
        total_timesteps=TOTAL_TIMESTEPS
    )
    # 组合多个 Callback
    callbacks = [academic_callback, ent_decay_callback]

    # 3. 初始化带有动作掩码的 PPO 算法
    # 这里使用的是针对 MultiInput (Dict) 的策略网络
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=linear_schedule(3e-4, 3e-5),
        n_steps=1024,           # 每收集 1024 步进行一次网络更新
        batch_size=128,
        ent_coef=0.01,          
        gamma=0.99,             # 折扣因子
        verbose=0,              # 关闭框架自带的杂乱日志，用 Callback 代替
        tensorboard_log=None    # 关闭 Tensorboard，使用 CSV
    )
    
    # 4. 启动学习过程
    print(f"[*] 设定总训练步数: {TOTAL_TIMESTEPS} Steps")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True       # 开启进度条
    )
    
    # 5. 保存最终模型权重
    model.save(MODEL_PATH)
    print(f"💾 模型权重已保存至: {MODEL_PATH}.zip")

if __name__ == "__main__":
    main()