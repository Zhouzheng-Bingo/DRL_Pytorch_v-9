import gym
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
# 导入您自定义的环境
from env_RL import TaskOffloadingEnv


class CustomPPOCallback(BaseCallback):
    def __init__(self, eval_env, check_freq, log_dir):
        super(CustomPPOCallback, self).__init__()
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_actions = []

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            obs = self.eval_env.reset()
            episode_rewards = []
            episode_actions = []
            for _ in range(10):
                done, ep_reward = False, 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.eval_env.step(action)
                    ep_reward += reward
                    episode_actions.append(action)
                episode_rewards.append(ep_reward)

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)

            self.logger.record('evaluation/mean_reward', mean_reward)
            self.logger.record('evaluation/std_reward', std_reward)

            for i in range(self.eval_env.action_space.n):
                self.logger.record(f'actions/action_{i}', episode_actions.count(i) / len(episode_actions))

        return True


if __name__ == '__main__':
    # 创建环境
    env = TaskOffloadingEnv(alpha=0.5)
    env = DummyVecEnv([lambda: env])

    eval_env = DummyVecEnv([lambda: TaskOffloadingEnv(alpha=0.7)])

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path='./logs/',
                                 log_path='./logs/',
                                 eval_freq=1000)

    custom_ppo_callback = CustomPPOCallback(eval_env, check_freq=1000, log_dir="./tensorboard_logs/")

    model.learn(total_timesteps=200000, callback=[eval_callback, custom_ppo_callback])

    # 保存模型
    model.save("ppo_task_offloading")

    # 加载模型
    loaded_model = PPO.load("ppo_task_offloading")

    # 测试模型
    obs = env.reset()
    total_reward = 0
    for _ in range(1000):
        action, _ = loaded_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            obs = env.reset()
            print(f"Total Reward: {total_reward}")
            total_reward = 0
