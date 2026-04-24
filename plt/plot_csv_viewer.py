import matplotlib.pyplot as plt
import pandas as pd
import os

def limit_data_jump(data, max_change=15, window_size=3):
    """
    限制数据的突然跳变，使曲线更平滑 (复用了你的超参数绘图脚本逻辑)
    """
    if not data or len(data) == 0:
        return data
    
    limited_data = [data[0]]  # 第一个点保持不变
    for i in range(1, len(data)):
        window_start = max(0, len(limited_data) - window_size)
        window_values = limited_data[window_start:]  
        reference_value = sum(window_values) / len(window_values)  
        
        current_value = data[i]
        change = current_value - reference_value
        
        # 如果变化超过阈值，则限制变化幅度
        if abs(change) > max_change:
            if change > 0:
                limited_value = reference_value + max_change
            else:
                limited_value = reference_value - max_change
            limited_data.append(limited_value)
        else:
            limited_data.append(current_value)
            
    return limited_data

def plot_training_metrics(csv_path="ppo_macro_training_metrics.csv"):
    # 1. 读取CSV数据
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        print(f"成功加载数据！共读取到 {len(df)} 轮(Episodes)数据。")
    except Exception as e:
        print(f"读取CSV失败，请检查文件路径: {e}")
        return

    # 提取横坐标 (Episode) 和 纵坐标 (Total_Reward)
    episodes = df['Episode'].tolist()
    raw_rewards = df['Total_Reward'].tolist()
    
    # 应用跳变限制以平滑曲线
    # max_change 可以根据你的 Reward 波动幅度自行调大或调小
    limited_rewards = limit_data_jump(raw_rewards, max_change=15, window_size=3)

    # 2. 绘制图表 (完美复现你的绘图风格)
    plt.figure(figsize=(10, 6))
    
    # 绘制处理后的曲线，使用之前的莫兰迪蓝色
    plt.plot(episodes, limited_rewards, 
            color='#6fa8dc',         
            label='Macro Agent (Total Reward)',
            linewidth=1.5,
            alpha=0.8)
    
    # 设置图表属性
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Macro Agent Training Learning Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')  # 虚线网格
    plt.tight_layout()
    
    # 3. 核心修改：弹出交互式窗口供你查看，并禁用自动保存图片
    print("正在打开图表查看器，关闭图表窗口后程序将自动结束...")
    plt.show()

if __name__ == '__main__':
    # 获取当前脚本所在目录 (training)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 向上退一级到根目录，然后进入 logs 文件夹
    csv_path = os.path.join(current_dir, '..', 'logs', 'ppo_macro_metrics_20260424_131908.csv')
    
    # 也可以打印出来确认一下路径对不对
    print(f"正在读取目标日志: {os.path.normpath(csv_path)}")
    
    plot_training_metrics(csv_path)