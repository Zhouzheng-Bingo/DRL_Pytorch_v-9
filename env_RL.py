import numpy as np
import gym
from RcnnPytorch.per_layer_time import partition_recommendations

# t
# latency_edge = [10 + i*2 for i in range(25)]
latency_edge = \
    [0.002274322509765625, 0.05997405529022217, 0.050271587371826174, 0.029629735946655272,
     0.02869260311126709, 0.017710041999816895, 0.0184151554107666, 0.011902337074279784,
     0.012034716606140137, 0.008485541343688965, 0.008024764060974122, 0.005768003463745117,
     0.006492080688476562, 0.005338711738586426, 0.004825563430786133, 0.004467272758483886,
     0.004373922348022461, 0.0036278438568115236, 0.005011301040649414, 0.004368014335632324,
     0.001481480598449707, 0.0042125463485717775, 0.0028664302825927735, 0.0031097793579101564,
     0.00032026290893554687]
# t_
# latency_server = [50 - i for i in range(25)]
latency_server = \
    [9.984970092773438e-05, 0.004100675582885742, 0.002580385208129883, 0.001679844856262207,
     0.001979794502258301, 0.0014796495437622071, 0.0016596317291259766, 0.00118133544921875,
     0.0012986087799072266, 0.0009001541137695312, 0.0010400533676147461, 0.0008999967575073242,
     0.0010000324249267577, 0.0008200836181640625, 0.0009798955917358397, 0.0007816505432128906,
     0.0008580923080444336, 0.0008201265335083008, 0.0009300851821899414, 0.000800185203552246,
     0.0003800058364868164, 0.0005401086807250977, 0.0005399370193481445, 0.0006599569320678711,
     5.9995651245117184e-05]
# data_t
# data_transmission = [2, 4, 6, 7, 10, 12, 15, 17, 18, 20, 21, 22, 23, 24, 24.5, 25, 25.5, 25.8, 26, 26.1, 26.2, 26.3, 26.4, 26.5, 26.6]
data_transmission_t = \
    [6.70242310e+00, 6.11328125e+01, 3.05664062e+01, 3.05664062e+01, 1.53320312e+01,
     1.53320312e+01, 7.71484375e+00, 7.71484375e+00, 3.90625000e+00, 3.90625000e+00,
     1.95312500e+00, 1.95312500e+00, 9.76562500e-01, 9.76562500e-01, 4.88281250e-01,
     4.88281250e-01, 2.92968750e-01, 2.92968750e-01, 1.95312500e-01, 1.95312500e-01,
     7.62939453e-04, 4.57763672e-02, 9.15527344e-02, 1.83105469e-01, 7.62939453e-04]
# data_t_
data_transmission_t_ = \
    [5.36193848e+00, 4.89062500e+01, 2.44531250e+01, 2.44531250e+01,
     1.22656250e+01, 1.22656250e+01, 6.17187500e+00, 6.17187500e+00,
     3.12500000e+00, 3.12500000e+00, 1.56250000e+00, 1.56250000e+00,
     7.81250000e-01, 7.81250000e-01, 3.90625000e-01, 3.90625000e-01,
     2.34375000e-01, 2.34375000e-01, 1.56250000e-01, 1.56250000e-01,
     6.10351562e-04, 3.66210938e-02, 7.32421875e-02, 1.46484375e-01,
     6.10351562e-04]

# throughput_edge = [15 - i*0.5 for i in range(25)]
# throughput_edge = \
#     [3862688.7621603487, 19021274.39551297, 13976966.488004338, 12883229.126237411,
#      62748446.4772795, 335078.22512172756, 201147.4180277644, 341278.77542362927,
#      178443.2029448546, 289101.5165324247, 139015.9324152796, 215083.80950931073,
#      106358.96480910413, 150844.82511560526, 79753.12359804267, 110956.937530071,
#      59148.98757829669, 71927.46467740077, 53050.799905928296, 57305.65690528417,
#      228.62774426073665, 16538.749286947925, 23945.877333373934, 54944.874617783746,
#      675.0004023315921]
# # throughput_server = [5 + i for i in range(25)]
# throughput_server = \
#     [87982236.4851958, 148355327.10273772, 74201246.74608992, 60706991.70393705,
#      334957610.80909234, 4900655.902620312, 3918794.7474433887, 6019603.514160249,
#      2586126.9915436916, 3460278.8353394237, 1542510.8806206004, 2167039.0080082663,
#      985670.2186253846, 1421978.7261092511, 615353.0385347179, 711113.6730898629,
#      383987.5492318403, 468244.9344125035, 261252.3233689702, 327512.08608867525,
#      1165.3757883915423, 73159.44253545202, 129020.4405981964, 299930.56474247813,
#      2631.538529105442]

# 计算最小值和最大值
min_latency = min(min(latency_edge), min(latency_server))
max_latency = max(max(latency_edge), max(latency_server))
# min_throughput = min(min(throughput_edge), min(throughput_server))
# max_throughput = max(max(throughput_edge), max(throughput_server))

if max_latency == min_latency:
    raise ValueError("Max and min latencies are the same. This will cause division by zero.")
# if max_throughput == min_throughput:
#     raise ValueError("Max and min throughputs are the same. This will cause division by zero.")


class TaskOffloadingEnv(gym.Env):
    def __init__(self, alpha=0.5):
        super(TaskOffloadingEnv, self).__init__()
        self.alpha = alpha

        # self.compute_requirements = [i * 10 for i in range(1, 26)]  # example compute requirements
        # self.data_transmission_requirements = [i for i in range(1, 26)]  # example data transmission requirements
        self.compute_requirements = \
            [25, 24, 6, 4, 1,
             8, 15, 7, 20, 10,
             16, 15, 23, 10, 2,
             25, 13, 14, 18, 20,
             16, 17, 5, 9, 1]
        self.data_transmission_requirements = \
            [25, 21, 19, 21, 24,
             17, 10, 18, 5, 15,
             9, 10, 2, 15, 23,
             0, 12, 11, 7, 5,
             9, 8, 20, 16, 24]
        self.action_space = gym.spaces.Discrete(2)  # Either edge or server for each task
        self.observation_space = gym.spaces.Box(low=0, high=250, shape=(6,), dtype=np.float32)  # As defined above

        self.current_task = 0
        self.previous_action = -1

        self.actions_taken = []

    def reset(self):
        self.current_task = 0
        self.previous_action = -1
        self.actions_taken = []  # Reset the actions taken list
        initial_state = [
            self.compute_requirements[self.current_task],
            self.data_transmission_requirements[self.current_task],
            self.current_task, 25 - self.current_task, 0,
            self.previous_action
        ]  # Last value is for previous action
        return np.array(initial_state)

    # def critic_evaluate(self):
    #     if not self.actions_taken:
    #         return 0
    #     critic_recommendations = partition_recommendations(2, latency_edge, latency_server, data_transmission_t, data_transmission_t_)
    #     print("Critic's partition recommendation:", critic_recommendations)
    #     matched_decisions = sum([1 if act == rec else 0 for act, rec in zip(self.actions_taken, critic_recommendations)])
    #     critic_reward = matched_decisions / len(self.actions_taken)
    #     return critic_reward

    def critic_evaluate(self):
        if not self.actions_taken:
            return 0
        critic_recommendations = partition_recommendations(2, latency_edge, latency_server, data_transmission_t,
                                                           data_transmission_t_)
        # Convert the values in critic_recommendations to integers
        critic_recommendations = [int(val) for val in critic_recommendations]

        # Convert all the values in self.actions_taken to integers
        self.actions_taken = [int(act) for act in self.actions_taken]

        print("Actions taken by the agent:", self.actions_taken)  # 打印智能体的行动
        print("Critic's partition recommendation:", critic_recommendations)  # 打印critic的建议
        matched_decisions = sum(
            [1 if act == rec else 0 for act, rec in zip(self.actions_taken, critic_recommendations)])

        print("Number of matched decisions:", matched_decisions)  # 打印匹配的数量
        critic_reward = matched_decisions  # 直接返回匹配任务的数量
        return critic_reward

    def step(self, action):
        if self.current_task == 25:
            # All tasks are done
            return self.reset()

        # Add the current action to the actions_taken list
        self.actions_taken.append(action)

        if action == 0:  # Execute on edge
            latency = latency_edge[self.current_task] + data_transmission_t[self.current_task]
            # throughput = throughput_edge[self.current_task]
        else:  # Execute on server
            latency = latency_server[self.current_task] + data_transmission_t_[self.current_task]
            # throughput = throughput_server[self.current_task]

        # Use max function to get throughput as the larger of the two latencies
        throughput = max(sum(latency_edge[:self.current_task + 1]) + data_transmission_t[self.current_task],
                         sum(latency_server[self.current_task:]) + data_transmission_t_[self.current_task])

        normalized_latency = (latency - min_latency) / (max_latency - min_latency)
        # normalized_throughput = (throughput - min_throughput) / (max_throughput - min_throughput)
        normalized_throughput = (throughput - min_latency) / (max_latency - min_latency)

        critic_weight = 1.0  # Adjust this based on the importance you want to give to the critic's recommendations

        # Original reward based on latency and throughput
        reward = -self.alpha * np.log(normalized_latency + 1e-3) + \
                 (1 - self.alpha) * np.log(normalized_throughput + 1e-3)
        print("Reward:", reward)
        self.previous_action = action
        # self.state.append(action[0])
        # Add the critic's evaluation to the reward
        critic_reward = self.critic_evaluate()
        print("Critic's reward:", critic_reward)
        normalized_critic_reward = (critic_reward - 0) / (1 - 0)  # Since min is 0 and max is 1
        reward += critic_weight * normalized_critic_reward
        print("Reward after adding critic's reward:", reward)
        self.current_task += 1

        if self.current_task == 25:
            done = True
        else:
            done = False

        next_state = [self.compute_requirements[self.current_task] if self.current_task < 25 else 0,
                      self.data_transmission_requirements[self.current_task] if self.current_task < 25 else 0,
                      self.current_task, 25 - self.current_task, action,
                      self.previous_action]

        return np.array(next_state), reward, done, {}

    def render(self, mode="human"):
        pass


if __name__ == "__main__":
    # Sample usage
    env = TaskOffloadingEnv(alpha=0.7)

    # Reset environment
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = np.random.choice([0, 1])  # Random action for testing
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Total Reward after completing all tasks: {total_reward}")
