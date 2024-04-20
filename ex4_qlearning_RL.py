import numpy as np

num_episodes = 1000  # 训练周期数量
max_steps_per_episode = 25  # 每个周期的最大步数，因为任务总数为25

env = TaskOffloadingEnv(alpha=0.7)  # 创建环境实例

for episode in range(num_episodes):
    state = env.reset()  # 重置环境
    done = False
    total_rewards = 0

    for step in range(max_steps_per_episode):
        # Epsilon-greedy action selection
        exploration_rate_threshold = np.random.uniform(0, 1)
        if exploration_rate_threshold > env.epsilon:
            action = np.argmax(env.q_table[state[2]])  # 利用（选择最优动作）
        else:
            action = env.action_space.sample()  # 探索（随机选择动作）

        next_state, reward, done, _ = env.step(action)  # 执行动作并观察结果

        # 更新Q表
        old_value = env.q_table[state[2], action]
        next_max = np.max(env.q_table[next_state[2]])
        new_value = old_value + env.learning_rate * (reward + env.gamma * next_max - old_value)
        env.q_table[state[2], action] = new_value

        state = next_state  # 进入新状态
        total_rewards += reward

        if done:
            break

    # 减少epsilon（探索率），减少探索行为，增加利用
    env.epsilon = min(env.epsilon * 0.99, 0.01)

    if episode % 100 == 0:
        print(f"Episode: {episode}, Total reward: {total_rewards}, Epsilon: {env.epsilon}")

print("Training finished.")
