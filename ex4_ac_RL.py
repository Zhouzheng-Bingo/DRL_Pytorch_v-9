import torch
import numpy as np
from ex4_env_RL import TaskOffloadingEnv  # Make sure to integrate the actor-critic methods into this import

# Setting up the environment
env = TaskOffloadingEnv(alpha=0.5)
state = env.reset()

# Hyperparameters
epochs = 1000
gamma = 0.99  # Discount factor
learning_rate = 1e-3

# Initialize actor and critic from the modified TaskOffloadingEnv
actor = env.actor
critic = env.critic
optimizer_actor = torch.optim.Adam(actor.parameters(), lr=learning_rate)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = actor(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        next_state, reward, done, _ = env.step(action.item())
        total_reward += reward

        # Get the value from the critic
        state_value = critic(state_tensor)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        next_state_value = critic(next_state_tensor)

        # Calculate TD Error
        target = reward + (gamma * next_state_value * (1 - int(done)))
        delta = target - state_value

        # Actor loss (policy gradient)
        actor_loss = -dist.log_prob(action) * delta.detach()

        # Critic loss (MSE)
        critic_loss = delta.pow(2)

        # Backpropagation
        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

        state = next_state

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Total Reward: {total_reward}')

# You can add evaluation code and plotting similar to your DQN example to compare performance
