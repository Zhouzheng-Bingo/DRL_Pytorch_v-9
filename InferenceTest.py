import numpy as np
import matplotlib.pyplot as plt
import time
from env_RL import TaskOffloadingEnv

if __name__ == '__main__':

    # 假设模拟的TaskOffloadingEnv类已经正确定义，并可以正常工作
    env = TaskOffloadingEnv(alpha=0.7)
    num_episodes = 10  # 模拟10个episode
    episode_lengths = 25  # 每个episode有25个决策步骤

    # 存储每个决策步骤的推断时间
    all_inference_times = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            start_time = time.time()  # 开始计时
            action = np.random.choice([0, 1])  # 随机选择动作
            state, reward, done, _ = env.step(action)
            end_time = time.time()  # 结束计时
            inference_time = (end_time - start_time) * 1000  # 计算推断时间，转换为毫秒
            all_inference_times.append(inference_time)

    # 将推断时间数据转换为Numpy数组
    inference_times = np.array(all_inference_times).reshape(num_episodes, episode_lengths)

    # 计算每个步骤的平均推断时间
    average_inference_times = np.mean(inference_times, axis=0)

    # 绘制每个步骤的平均推断时间
    plt.figure(figsize=(10, 6))
    plt.plot(average_inference_times, label='平均推断时间', marker='o')
    plt.xlabel('决策步骤')
    plt.ylabel('推断时间 (毫秒)')
    plt.title('每个决策步骤的平均推断时间')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 返回平均推断时间以便进一步使用
    average_inference_times