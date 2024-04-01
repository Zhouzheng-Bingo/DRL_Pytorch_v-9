import gym
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_schedule_fn, linear_schedule
import matplotlib.pyplot as plt

import os

from stable_baselines3.dqn.policies import DQNPolicy, MlpPolicy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from env_RL import TaskOffloadingEnv

# 确保matplotlib使用TkAgg后端
matplotlib.use('TkAgg')


# 定义自定义特征提取器
class CustomNetworkReLU(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 64):
        super(CustomNetworkReLU, self).__init__(observation_space, features_dim)
        # 定义网络结构
        self.network = nn.Sequential(
            nn.Linear(observation_space.shape[0], features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


# 定义自定义策略
# class CustomMlpPolicy(MlpPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomMlpPolicy, self).__init__(*args, **kwargs)
#
#         # 定义自定义的网络结构
#         self.features_extractor = CustomNetworkReLU(self.features_dim)


# 定义自定义回调
class CustomCallback(BaseCallback):
    def __init__(self, eval_env, check_freq, log_dir):
        super(CustomCallback, self).__init__()
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_actions = []
        self.all_actions = []  # 初始化空列表以收集所有动作

    def _on_step(self):
        # This method will be called by the model after each call to `env.step()`.
        if self.n_calls % self.check_freq == 0:
            obs = self.eval_env.reset()
            episode_rewards = []
            episode_actions = []
            for _ in range(10):  # Evaluate over 10 episodes
                done, ep_reward = False, 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.eval_env.step(action)
                    ep_reward += reward
                    episode_actions.append(action)
                    self.all_actions.append(action[0])
                episode_rewards.append(ep_reward)

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)

            # Log the results in TensorBoard
            self.logger.record('evaluation/mean_reward', mean_reward)
            self.logger.record('evaluation/std_reward', std_reward)

            for i in range(self.eval_env.action_space.n):
                self.logger.record(f'actions/action_{i}', episode_actions.count(i) / len(episode_actions))

            # # Check for improvement
            # if mean_reward > self.best_mean_reward:
            #     self.best_mean_reward = mean_reward
            #     self.no_improvement_steps = 0
            # else:
            #     self.no_improvement_steps += 1
            #
            # if self.no_improvement_steps >= 5:
            #     print("No improvement for 5 consecutive checks. Stopping training...")
            #     return False

        return True


if __name__ == '__main__':
    # 初始化环境
    env = TaskOffloadingEnv(alpha=0.5)
    env = DummyVecEnv([lambda: env])

    policy_kwargs = dict(
        features_extractor_class=CustomNetworkReLU,
        features_extractor_kwargs=dict(features_dim=64),
    )

    # 创建评估环境
    eval_env = DummyVecEnv([lambda: TaskOffloadingEnv(alpha=0.7)])

    # 定义动态学习率
    learning_rate_schedule = linear_schedule(1e-4, 1e-6, 100000)

    # 使用自定义策略创建DQN模型
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/",
                policy_kwargs=policy_kwargs,
                learning_rate=1e-4,
                exploration_final_eps=0.01,
                exploration_fraction=0.2)

    # 定义评估回调
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=1000)
    custom_callback = CustomCallback(eval_env, check_freq=1000, log_dir="./tensorboard_logs/")

    # 训练智能体
    model.learn(total_timesteps=1000000, callback=[eval_callback, custom_callback])

    # 保存训练好的模型
    model.save("dqn_task_offloading")

    # 加载训练好的模型进行评估
    model = DQN.load("dqn_task_offloading")

    # 测试训练好的智能体
    # Evaluate the trained agent's performance
    num_episodes = 10
    average_reward = 0
    all_actions = []

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            all_actions.append(action[0])
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        average_reward += episode_reward

    average_reward /= num_episodes
    print(f"Average reward over {num_episodes} episodes: {average_reward}")

    # Test the trained agent and record its actions
    obs = env.reset()
    done = False
    actions_taken = []

    while not done:
        action, _ = model.predict(obs)
        actions_taken.append(action[0])
        obs, _, done, _ = env.step(action)
    print("Actions taken by the trained agent:", actions_taken)

    # Action distribution after evaluation
    plt.hist(custom_callback.all_actions, bins=range(env.action_space.n + 1), align='left', rwidth=0.8)
    plt.xlabel('Actions')
    plt.ylabel('Frequency')
    plt.title('Action Distribution')
    plt.show()

    # Plot the results
    results = np.load('logs/evaluations.npz')
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(results['timesteps'], results['results'])
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward')
    plt.title('Training Convergence Plot')

    # Plot the action distribution
    plt.subplot(1, 2, 2)
    plt.hist(all_actions, bins=range(len(np.unique(all_actions)) + 1), align='left', rwidth=0.8)
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.title('Action Distribution during Evaluation')

    plt.tight_layout()
    plt.show()
