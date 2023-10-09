import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env_RL import TaskOffloadingEnv

# Initialize environment
env = TaskOffloadingEnv(alpha=0.7)
env = DummyVecEnv([lambda: env])  # DQN requires a vectorized environment

# Define the DQN agent
# model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_tensorboard/")
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=None)

# Train the agent
model.learn(total_timesteps=1000)

# Save the trained model
model.save("dqn_task_offloading")

# Load the trained model for evaluation
model = DQN.load("dqn_task_offloading")

# Evaluate the trained agent's performance
num_episodes = 10
average_reward = 0
for episode in range(num_episodes):
    obs = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
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
