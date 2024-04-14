import time
import psutil
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from env_RL import TaskOffloadingEnv

import matplotlib
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt

# 定义监控资源使用情况的函数
def monitor_resources(interval=5.0):
    cpu_usage = psutil.cpu_percent(interval=interval)
    memory_usage = psutil.virtual_memory().percent
    return cpu_usage, memory_usage

# 定义执行模型推理的函数
def infer_model(model, num_episodes, env, monitor_interval_steps=100):
    inference_times, cpu_usages, memory_usages = [], [], []
    total_rewards = []
    inference_step_counter = 0

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            start_time = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            duration = time.time() - start_time

            episode_reward += reward
            inference_times.append(duration)

            # 每一步后都监控资源使用情况
            cpu_usage, memory_usage = monitor_resources(interval=0.0)  # Set interval to 0 for non-blocking call
            cpu_usages.append(cpu_usage)
            memory_usages.append(memory_usage)

        total_rewards.append(episode_reward)

    return total_rewards, inference_times, cpu_usages, memory_usages

if __name__ == '__main__':
    eval_env = DummyVecEnv([lambda: TaskOffloadingEnv(alpha=0.7)])
    model = DQN.load("dqn_task_offloading")

    # 执行模型推理并监控性能
    total_rewards, inference_times, cpu_usages, memory_usages = infer_model(model, 10, eval_env)

    # 保存推理性能数据
    inference_data = {
        "Inference Times": inference_times,
        "CPU Usage (%)": cpu_usages,
        "Memory Usage (%)": memory_usages
    }
    df_inference = pd.DataFrame(inference_data)
    df_inference.to_csv("inference_performance_data.csv", index=False)

    # 选择性地绘制结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(inference_times, label='Inference Time per Step')
    plt.legend()
    plt.title('Inference Time per Step')
    plt.xlabel('Episode')
    plt.ylabel('Time (s)')

    plt.subplot(1, 2, 2)
    plt.plot(cpu_usages, label='Inference CPU Usage')
    plt.legend()
    plt.title('CPU Usage during Inference')
    plt.xlabel('Episode')
    plt.ylabel('Usage (%)')

    plt.tight_layout()
    plt.savefig("inference_performance_plots.png")
    plt.show()