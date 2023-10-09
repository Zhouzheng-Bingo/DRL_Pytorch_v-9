import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from env_RL import TaskOffloadingEnv

# Initialize environment
env = TaskOffloadingEnv(alpha=0.7)
env = DummyVecEnv([lambda: env])  # DQN requires a vectorized environment

# Create evaluation environment
eval_env = DummyVecEnv([lambda: TaskOffloadingEnv(alpha=0.7)])

# Define the DQN agent
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=None)

# Define the evaluation callback
eval_callback = EvalCallback(eval_env,
                             best_model_save_path='./logs/',
                             log_path='./logs/',
                             eval_freq=500)

# Train the agent with callback
model.learn(total_timesteps=100000, callback=eval_callback)

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

# Plot the results
results = np.load('logs/evaluations.npz')
plt.plot(results['timesteps'], results['results'])
plt.xlabel('Timesteps')
plt.ylabel('Average Reward')
plt.title('Training Convergence Plot')
plt.show()
