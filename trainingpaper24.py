import numpy as np
import gym

from ex4_paper24_RL import TaskOffloadingEnv


class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.epsilon_min = epsilon_min  # Minimum value for epsilon
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        target = reward + (self.gamma * np.max(self.q_table[next_state]) * (1 - done))
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_agent(episodes=1000):
    env = TaskOffloadingEnv(alpha=0.5)  # Initialize the environment
    agent = QLearningAgent(state_size=25, action_size=2)  # Assuming there are 25 states and 2 actions

    for e in range(episodes):
        state = env.reset()
        state = int(state[2])  # Assuming the third index of the state array is the task index
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = int(next_state[2])  # Assuming the third index of the state array is the task index
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        print(f'Episode: {e+1}/{episodes}, Total reward: {total_reward}, Epsilon: {agent.epsilon:.2f}')

    return agent

if __name__ == "__main__":
    trained_agent = train_agent()
