import numpy as np
import matplotlib.pyplot as plt


class SimpleEnvironment:
    def __init__(self):
        self.state = 0

    def step(self, action):
        reward = np.random.normal(0, 1)
        return self.state, reward, False, {}

    def reset(self):
        return self.state


class LinearPolicy:
    def __init__(self, num_actions):
        self.weights = np.random.randn(num_actions)

    def select_action(self, state):
        action = np.dot(self.weights, state)
        return action


class LinearCritic:
    def __init__(self):
        self.weights = np.random.randn(1)

    def value(self, state):
        return np.dot(self.weights, state)


def actor_critic(env, policy, critic, episodes, timesteps):
    actor_losses = []
    critic_losses = []

    for _ in range(episodes):
        state = env.reset()
        for _ in range(timesteps):
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)

            value = critic.value(state)
            next_value = critic.value(next_state)

            td_error = reward + 0.99 * next_value - value
            critic_loss = td_error ** 2
            actor_loss = -td_error * action

            critic.weights += 0.01 * td_error
            policy.weights += 0.01 * actor_loss

            state = next_state

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            if done:
                break

    return actor_losses, critic_losses


env = SimpleEnvironment()
policy = LinearPolicy(num_actions=1)
critic = LinearCritic()

actor_losses, critic_losses = actor_critic(env, policy, critic, episodes=10, timesteps=1000000)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(actor_losses, label='Actor Loss')
plt.title('Actor Loss Over Time')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(critic_losses, label='Critic Loss')
plt.title('Critic Loss Over Time')
plt.legend()

plt.show()
