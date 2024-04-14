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
matplotlib.use('TkAgg')  # 使用TkAgg后端，你也可以尝试其他后端名称
import matplotlib.pyplot as plt

def monitor_resources(interval=1.0):
    cpu_usage = psutil.cpu_percent(interval=interval)
    memory_usage = psutil.virtual_memory().percent
    return cpu_usage, memory_usage

class TrainingCallback(EvalCallback):
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

if __name__ == '__main__':
    env = DummyVecEnv([lambda: TaskOffloadingEnv(alpha=0.5)])
    eval_env = DummyVecEnv([lambda: TaskOffloadingEnv(alpha=0.7)])

    model = DQN("MlpPolicy", env, verbose=1,
                tensorboard_log="./tensorboard_logs/",
                learning_rate=0.0001,
                exploration_final_eps=0.01,
                exploration_fraction=0.2)

    training_callback = TrainingCallback(eval_env, check_freq=1000, log_dir="./tensorboard_logs/")
    model.learn(total_timesteps=1000, callback=training_callback)

    # Save training performance data
    training_data = {
        "Training Times": training_callback.training_times,
        "CPU Usage (%)": training_callback.cpu_usages,
        "Memory Usage (%)": training_callback.memory_usages
    }
    df_training = pd.DataFrame(training_data)
    df_training.to_csv("training_performance_data.csv", index=False)

    # Optionally plot the results
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