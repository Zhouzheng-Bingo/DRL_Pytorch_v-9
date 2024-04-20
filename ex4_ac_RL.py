import torch
import numpy as np
from ex4_paper1_RL import TaskOffloadingEnv  # 确保已集成actor-critic方法


if __name__ == '__main__':

    # 开启异常检测
    torch.autograd.set_detect_anomaly(True)

    # 设置环境
    env = TaskOffloadingEnv(alpha=0.5)
    state = env.reset()

    # 超参数
    epochs = 1000
    gamma = 0.99  # 折扣因子
    learning_rate = 1e-3

    # 从修改过的TaskOffloadingEnv初始化actor和critic
    actor = env.actor
    critic = env.critic
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=learning_rate)

    # 训练循环
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

            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                next_state_value = critic(next_state_tensor)

            state_value = critic(state_tensor)
            target = reward + gamma * next_state_value * (1 - int(done))
            delta = target - state_value

            actor_loss = -dist.log_prob(action) * delta.detach()
            critic_loss = delta.pow(2)

            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            optimizer_actor.step()
            optimizer_critic.step()

            state = next_state

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Total Reward: {total_reward}')
