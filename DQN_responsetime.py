import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import linear_schedule
from stable_baselines3.common.callbacks import BaseCallback
from env_RL import TaskOffloadingEnv
import matplotlib
import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

matplotlib.use('TkAgg')  # 使用TkAgg后端，你也可以尝试其他后端名称
import matplotlib.pyplot as plt

'''
# 将下边这一段贴到stable_baselines3.common.utils.py中，否则导包stable_baselines3.common.utils.py会报错
# def linear_schedule(initial_value):
#     """
#     Linear learning rate schedule.
#     :param initial_value: (float) Initial learning rate.
#     :return: (function)
#     """
#     def func(progress_remaining):
#         """
#         Progress will decrease from 1 (beginning) to 0
#         :param progress_remaining: (float)
#         :return: (float)
#         """
#         return progress_remaining * initial_value
#
#     return func

def linear_schedule(initial_value, final_value, schedule_timesteps):
    """
    Linear learning rate schedule.
    :param initial_value: (float) Initial learning rate.
    :param final_value: (float) Final learning rate.
    :param schedule_timesteps: (int) Number of timesteps for the schedule.
    :return: (function)
    """
    def func(t):
        """
        Progress will decrease from 1 (beginning) to 0
        :param t: (int) Current timestep.
        :return: (float)
        """
        fraction = min(float(t) / schedule_timesteps, 1.0)
        return initial_value + fraction * (final_value - initial_value)

    return func
'''


class CustomCallback(BaseCallback):
    def __init__(self, eval_env, check_freq, log_dir):
        super(CustomCallback, self).__init__()
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_actions = []
        # self.best_mean_reward = -np.inf
        # self.no_improvement_steps = 0
        self.response_times = []  # To store response times for analysis
    def _on_step(self):
        # This method will be called by the model after each call to `env.step()`.
        if self.n_calls % self.check_freq == 0:
            obs = self.eval_env.reset()
            episode_rewards = []
            episode_actions = []
            episode_response_times = []  # Local storage for this evaluation
            for _ in range(10):  # Evaluate over 10 episodes
                done, ep_reward = False, 0
                while not done:
                    start_time = time.time()  # Start timing
                    action, _ = self.model.predict(obs, deterministic=True)
                    response_time = time.time() - start_time  # End timing
                    episode_response_times.append(response_time)
                    obs, reward, done, _ = self.eval_env.step(action)
                    ep_reward += reward
                    episode_actions.append(action)
                episode_rewards.append(ep_reward)
            self.response_times.extend(episode_response_times)  # Add to global list

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_response_time = np.mean(episode_response_times)  # Calculate mean response time

            # Log the results in TensorBoard
            self.logger.record('evaluation/mean_reward', mean_reward)
            self.logger.record('evaluation/std_reward', std_reward)
            self.logger.record('evaluation/mean_response_time', mean_response_time)

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
    # Initialize environment
    env = TaskOffloadingEnv(alpha=0.5)
    env = DummyVecEnv([lambda: env])  # DQN requires a vectorized environment

    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: TaskOffloadingEnv(alpha=0.7)])

    # Set up a dynamic learning rate using linear_schedule
    learning_rate_schedule = linear_schedule(1e-4, 1e-6, 100000)  # Modified
    # learning_rate_schedule = 1e-6
    # Define the DQN agent with dynamic learning rate and adjusted exploration strategy
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/",
                learning_rate=learning_rate_schedule,  # Modified
                exploration_final_eps=0.01,  # Added
                exploration_fraction=0.2)  # Modified

    # model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/",
    #             learning_rate=1e-3,  # Modified
    #             exploration_final_eps=0.02,  # Added
    #             exploration_fraction=0.3)  # Modified

    # 定义评估回调
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=1000)

    custom_callback = CustomCallback(eval_env, check_freq=1000, log_dir="./tensorboard_logs/")

    # Train the agent with callback
    model.learn(total_timesteps=1000000, callback=[eval_callback, custom_callback])

    # Save the trained model
    model.save("dqn_task_offloading")

    # Load the trained model for evaluation
    model = DQN.load("dqn_task_offloading")

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
