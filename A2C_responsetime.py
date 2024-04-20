import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from env_RL import TaskOffloadingEnv
import time

class CustomCallback(BaseCallback):
    def __init__(self, eval_env, check_freq, log_dir):
        super(CustomCallback, self).__init__()
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_actions = []
        self.response_times = []  # To store response times for analysis

    def _on_step(self):
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

        return True


if __name__ == '__main__':
    # Initialize environments
    num_envs = 4
    env = SubprocVecEnv([lambda: TaskOffloadingEnv(alpha=0.5) for _ in range(num_envs)])
    eval_env = DummyVecEnv([lambda: TaskOffloadingEnv(alpha=0.7)])

    # Define the A2C agent
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")

    # Define the evaluation callback
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path='./logs/',
                                 log_path='./logs/',
                                 eval_freq=1000)
    custom_callback = CustomCallback(eval_env, check_freq=1000, log_dir="./tensorboard_logs/")

    # Train the agent with callback
    model.learn(total_timesteps=1000000, callback=[eval_callback, custom_callback])

    # Save the trained model
    model.save("a2c_task_offloading")

    # Load the trained model for evaluation
    model = A2C.load("a2c_task_offloading")

    # Evaluate the trained agent's performance
    num_episodes = 10
    average_reward = 0
    all_actions = []

    eval_env = DummyVecEnv([lambda: TaskOffloadingEnv(alpha=0.7)])  # Use a non-vectorized env for evaluation

    for episode in range(num_episodes):
        obs = eval_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            all_actions.append(action[0])
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward
        average_reward += episode_reward

    average_reward /= num_episodes
    print(f"Average reward over {num_episodes} episodes: {average_reward}")

    # Test the trained agent and record its actions
    obs = eval_env.reset()
    done = False
    actions_taken = []

    while not done:
        action, _ = model.predict(obs)
        actions_taken.append(action[0])
        obs, _, done, _ = eval_env.step(action)
        eval_env.envs[0].actions_taken.append(action[0])
    print("Actions taken by the trained agent:", actions_taken)
