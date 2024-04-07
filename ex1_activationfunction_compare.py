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
class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 64, activation_fn: nn.Module = nn.ReLU):
        super(CustomNetwork, self).__init__(observation_space, features_dim)
        # 定义网络结构
        self.network = nn.Sequential(
            nn.Linear(observation_space.shape[0], features_dim),
            activation_fn(),
            nn.Linear(features_dim, features_dim),
            activation_fn()
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
    def __init__(self, eval_env, check_freq, log_dir, record_freq=1000, activation_fn_name='ReLU'):
        super(CustomCallback, self).__init__()
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_actions = []
        self.all_actions = []  # 初始化空列表以收集所有动作
        self.record_freq = record_freq  # 每record_freq步记录一次
        self.local_step = 0
        self.sum_rewards = 0.0
        self.activation_fn_name = activation_fn_name

    def _on_step(self):
        self.local_step += 1
        # 确保此处有方法获取奖励
        reward = self.locals["rewards"][0] if "rewards" in self.locals else 0
        self.sum_rewards += reward

        if self.local_step % self.record_freq == 0:
            # 计算自上次记录以来所有步骤的平均奖励
            avg_reward = self.sum_rewards / self.record_freq
            # self.logger.record('train/average_reward', avg_reward)
            self.logger.record(f'train/{self.activation_fn_name}_average_reward', avg_reward)
            # 重置计数器和奖励总和
            self.local_step = 0
            self.sum_rewards = 0.0

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
    activation_functions = {
        'ReLU': nn.ReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh
    }

    results = {}

    for name, activation_fn in activation_functions.items():
        # 初始化环境
        env = TaskOffloadingEnv(alpha=0.5)
        env = DummyVecEnv([lambda: env])

        policy_kwargs = dict(
            features_extractor_class=CustomNetwork,
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
        # custom_callback = CustomCallback(eval_env, check_freq=1000, log_dir="./tensorboard_logs/")
        custom_callback = CustomCallback(eval_env, check_freq=1000, log_dir="./tensorboard_logs/",
                                         activation_fn_name=name)
        # 训练智能体
        model.learn(total_timesteps=1000000, callback=[eval_callback, custom_callback])

        # 保存训练好的模型
        model.save(f"dqn_task_offloading_{name}")

