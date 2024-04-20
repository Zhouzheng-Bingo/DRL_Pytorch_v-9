import numpy as np
import pandas as pd
import time
from torch.utils.tensorboard import SummaryWriter  # 引入TensorBoard

np.random.seed(2)  # reproducible

N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 10   # maximum episodes
FRESH_TIME = 0.1    # fresh time for one move
TOTAL_TIMESTEPS = 1000000  # total timesteps

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    return table

def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R

def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATES-1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    writer = SummaryWriter('runs/q_learning_avg_std')
    q_table = build_q_table(N_STATES, ACTIONS)
    rewards = []
    timestep = 0
    for episode in range(MAX_EPISODES):
        S = 0
        is_terminated = False
        update_env(S, episode, 0)
        while not is_terminated and timestep < TOTAL_TIMESTEPS:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            rewards.append(R)
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            update_env(S, episode, timestep+1)
            timestep += 1

            if timestep % 10000 == 0:  # 记录平均奖励和标准差每10000个时间步
                avg_reward = np.mean(rewards)
                std_dev_reward = np.std(rewards)
                writer.add_scalar('Average Reward', avg_reward, timestep)
                writer.add_scalar('Standard Deviation of Reward', std_dev_reward, timestep)
                rewards = []  # reset rewards list for next recording interval

    writer.close()
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
