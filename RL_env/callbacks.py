import os
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

class AcademicMetricsCallback(BaseCallback):
    def __init__(self, save_path: str, save_freq_episodes: int = 50, verbose: int = 1):
        super(AcademicMetricsCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq_episodes = save_freq_episodes
        self.memory_buffer = [] 
        
        self.current_ep_metrics = {}
        self.current_ep_steps = 0
        self.current_ep_total_reward = 0.0
        self.episode_count = 0

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self) -> bool:
        info = self.locals["infos"][0] 
        step_reward = self.locals["rewards"][0]
        
        # 严格累计回合总奖励
        self.current_ep_total_reward += step_reward
        self.current_ep_steps += 1
        
        # 加入 CMDP 维度的指标收集
        for key, value in info.items():
            if key.startswith(("Metrics/", "Rewards/", "CMDP/")):
                if key not in self.current_ep_metrics:
                    self.current_ep_metrics[key] = 0.0
                self.current_ep_metrics[key] += value
        
        if self.locals["dones"][0]:
            self.episode_count += 1
            ep_summary = {"Episode": self.episode_count}
            
            # 分类进行平均化或累加
            for key, total_val in self.current_ep_metrics.items():
                # 列表1：这些物理状态指标需要求平均（表示整个回合的平均水平）
                if key in [
                    "Metrics/Avg_EndToEnd_Delay_ms", 
                    "Metrics/SLA_Success_Rate", 
                    "Metrics/Load_Variance",
                    "Metrics/Total_Cost",
                    "CMDP/Dynamic_Lambda_SLA"
                ]:
                    ep_summary[key] = total_val / self.current_ep_steps
                
                # 列表2：这些事件计数和奖励/惩罚必须保持总计（Sum）
                else:
                    ep_summary[key] = total_val
                
            # 存入正确的累计总奖励
            ep_summary["Total_Reward"] = self.current_ep_total_reward
            
            # ================= PPO 内部算法指标 =================
            if self.logger is not None:
                # 使用 logger.name_to_value 提取最近一次网络更新的 Loss
                ep_summary["Loss/Actor"] = self.logger.name_to_value.get("train/policy_gradient_loss", 0.0)
                ep_summary["Loss/Critic"] = self.logger.name_to_value.get("train/value_loss", 0.0)
                ep_summary["Loss/Entropy"] = self.logger.name_to_value.get("train/entropy_loss", 0.0)
                ep_summary["Loss/Approx_KL"] = self.logger.name_to_value.get("train/approx_kl", 0.0)
            # ==============================================================
            
            self.memory_buffer.append(ep_summary)
            
            if self.verbose > 0 and self.episode_count % 10 == 0:
                t_reward = ep_summary.get('Total_Reward', 0)
                delay = ep_summary.get('Metrics/Avg_EndToEnd_Delay_ms', 0)
                sla = ep_summary.get('Metrics/SLA_Success_Rate', 0)
                
                cost = ep_summary.get('Metrics/Total_Cost', 0)
                # 以下指标现在是回合总计，不再是“次/步”
                trunc_actions = ep_summary.get('Metrics/Truncated_Actions', 0)
                ai_thrash = ep_summary.get('Metrics/AI_Instance_Thrashing', 0)
                
                # 惩罚项现在展示的是回合总罚分，能直观反映模型多“疼”
                r_base = ep_summary.get('Rewards/R_base', 0)
                p_sla = ep_summary.get('Rewards/Penalty_SLA', 0)
                p_intent = ep_summary.get('Rewards/Penalty_Intent', 0)
                p_smooth = ep_summary.get('Rewards/Penalty_Smoothness', 0)
                
                # 新增输出拉格朗日乘子，观察其是否收敛
                lambda_sla = ep_summary.get('CMDP/Dynamic_Lambda_SLA', 0)

                report = (
                    f"\n[{self.episode_count:04d} 回合简报] 🏆总Reward: {t_reward:.2f} | ⏱️平均延迟: {delay:.1f}ms | ✅SLA满足: {sla:.2%}\n"
                    f" ├─ 📈 物理探测: 💰均成本: {cost:.1f} | ✂️违规截断: {trunc_actions:.0f}次/回合 | 📳AI频繁启停: {ai_thrash:.0f}次/回合\n"
                    f" └─ ⚖️ 奖励拆解: 🎯总基础R: {r_base:.2f} | 📉SLA总罚(λ={lambda_sla:.3f}): {p_sla:.2f} | 🚫意图总罚: {p_intent:.2f} | 🔄抖动总罚: {p_smooth:.2f}"
                )
                tqdm.write(report) 
            
            # 清空累计器，准备下一个 Episode
            self.current_ep_metrics = {}
            self.current_ep_steps = 0
            self.current_ep_total_reward = 0.0
            
            if self.episode_count % self.save_freq_episodes == 0:
                self._dump_to_csv()
                
        return True

    def _dump_to_csv(self):
        if len(self.memory_buffer) == 0: return
        df = pd.DataFrame(self.memory_buffer)
        write_header = not os.path.exists(self.save_path)
        df.to_csv(self.save_path, mode='a', index=False, header=write_header)
        if self.verbose > 1:
            print(f"[*] 已将 {len(self.memory_buffer)} 条回合数据安全落盘至 {self.save_path}")
        self.memory_buffer = []

    def _on_training_end(self) -> None:
        self._dump_to_csv()
        if self.verbose > 0:
            print(f"✅ 训练结束。所有监控指标已完整存入: {self.save_path}")