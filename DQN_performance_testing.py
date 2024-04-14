import numpy as np
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from env_RL import TaskOffloadingEnv

import matplotlib

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

matplotlib.use('TkAgg')  # 使用TkAgg后端，你也可以尝试其他后端名称
import matplotlib.pyplot as plt


def monitor_resources(interval=1.0):
    cpu_usage = psutil.cpu_percent(interval=interval)
    memory_usage = psutil.virtual_memory().percent
    return cpu_usage, memory_usage


class CustomCallback(EvalCallback):
    def __init__(self, eval_env, check_freq, log_dir):
        super().__init__(eval_env, best_model_save_path='./logs/',
                         log_path=log_dir, eval_freq=check_freq)
        self.training_times = []
        self.cpu_usages = []
        self.memory_usages = []

    def _on_step(self):
        start_time = time.time()
        super()._on_step()
        duration = time.time() - start_time
        cpu_usage, memory_usage = monitor_resources()

        self.training_times.append(duration)
        self.cpu_usages.append(cpu_usage)
        self.memory_usages.append(memory_usage)

        return True


def infer_model(model, num_episodes, env):
    inference_times, cpu_usages, memory_usages = [], [], []
    total_rewards = []

    for _ in range(num_episodes):
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

            cpu_usage, memory_usage = monitor_resources()
            cpu_usages.append(cpu_usage)
            memory_usages.append(memory_usage)

        total_rewards.append(episode_reward)

    return total_rewards, inference_times, cpu_usages, memory_usages


if __name__ == '__main__':
    env = DummyVecEnv([lambda: TaskOffloadingEnv(alpha=0.5)])
    eval_env = DummyVecEnv([lambda: TaskOffloadingEnv(alpha=0.7)])

    model = DQN("MlpPolicy", env, verbose=1,
                tensorboard_log="./tensorboard_logs/",
                learning_rate=0.0001,
                exploration_final_eps=0.01,
                exploration_fraction=0.2)

    custom_callback = CustomCallback(eval_env, check_freq=1000, log_dir="./tensorboard_logs/")
    model.learn(total_timesteps=1000000, callback=custom_callback)

    total_rewards, inference_times, cpu_usages, memory_usages = infer_model(model, 10, eval_env)

    # Save training performance data
    training_data = {
        "Training Times": custom_callback.training_times,
        "CPU Usage (%)": custom_callback.cpu_usages,
        "Memory Usage (%)": custom_callback.memory_usages
    }
    df_training = pd.DataFrame(training_data)
    df_training.to_csv("training_performance_data.csv", index=False)

    # Save inference performance data
    inference_data = {
        "Inference Times": inference_times,
        "CPU Usage (%)": cpu_usages,
        "Memory Usage (%)": memory_usages
    }
    df_inference = pd.DataFrame(inference_data)
    df_inference.to_csv("inference_performance_data.csv", index=False)

    # Optionally plot the results
    plt.figure(figsize=(15, 5))
    # Plot for training and inference times
    plt.subplot(1, 3, 1)
    plt.plot(df_training['Training Times'], label='Training Time per Step')
    plt.plot(df_inference['Inference Times'], label='Inference Time per Step')
    plt.legend()
    plt.title('Time per Step')
    plt.xlabel('Step')
    plt.ylabel('Time (s)')

    # Plot for CPU usage
    plt.subplot(1, 3, 2)
    plt.plot(df_training['CPU Usage (%)'], label='Training CPU Usage')
    plt.plot(df_inference['CPU Usage (%)'], label='Inference CPU Usage')
    plt.legend()
    plt.title('CPU Usage')
    plt.xlabel('Step')
    plt.ylabel('Usage (%)')

    # Plot for Memory usage
    plt.subplot(1, 3, 3)
    plt.plot(df_training['Memory Usage (%)'], label='Training Memory Usage')
    plt.plot(df_inference['Memory Usage (%)'], label='Inference Memory Usage')
    plt.legend()
    plt.title('Memory Usage')
    plt.xlabel('Step')
    plt.ylabel('Usage (%)')
    # plt.subplot(1, 2, 1)
    # plt.plot(df_training['Training Times'], label='Training Time per Step')
    # plt.plot(df_inference['Inference Times'], label='Inference Time per Step')
    # plt.legend()
    # plt.title('Time per Step')
    # plt.xlabel('Step')
    # plt.ylabel('Time (s)')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(df_training['CPU Usage (%)'], label='Training CPU Usage')
    # plt.plot(df_inference['CPU Usage (%)'], label='Inference CPU Usage')
    # plt.legend()
    # plt.title('CPU Usage')
    # plt.xlabel('Step')
    # plt.ylabel('Usage (%)')

    plt.tight_layout()
    plt.savefig("performance_plots.png")
    plt.show()
