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

# latency_edge = \
#     [0.0017840862274169922, 0.055511608123779296, 0.046255025863647464, 0.026398744583129883,
#      0.02550499439239502, 0.015221624374389649, 0.015032873153686524, 0.009385347366333008,
#      0.009438247680664062, 0.006461811065673828, 0.00648031234741211, 0.004691257476806641,
#      0.005340361595153808, 0.0043770503997802735, 0.004388885498046875, 0.003366870880126953,
#      0.00394927978515625, 0.0031490278244018554, 0.0038029098510742186, 0.003608689308166504,
#      0.0013907146453857422, 0.0017036724090576172, 0.001566634178161621, 0.0017315959930419922,
#      0.00022295475006103515]

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

# latency_server = \
#     [0.00010004043579101563, 0.003901114463806152, 0.0025202274322509766, 0.0017399787902832031,
#      0.0018593263626098633, 0.0015398502349853516, 0.0014998579025268554, 0.00110015869140625,
#      0.0011598491668701172, 0.0009201240539550782, 0.0009399843215942383, 0.0007199668884277344,
#      0.0008999347686767578, 0.0007199382781982422, 0.0009598875045776367, 0.0006600189208984375,
#      0.0008703041076660157, 0.000800013542175293, 0.0008199596405029297, 0.0006801223754882812,
#      0.0004999971389770508, 0.0006600522994995117, 0.0005199527740478516, 0.0007000064849853516,
#      0.0]

# data_t
# data_transmission = [2, 4, 6, 7, 10, 12, 15, 17, 18, 20, 21, 22, 23, 24, 24.5, 25, 25.5, 25.8, 26, 26.1, 26.2, 26.3, 26.4, 26.5, 26.6]

# 除以2
# data_transmission_t = \
#     [3.35121155, 30.56640625, 15.283203125, 15.283203125, 7.666015625,
#      7.666015625, 3.857421875, 3.857421875, 1.953125, 1.953125, 0.9765625,
#      0.9765625, 0.48828125, 0.48828125, 0.244140625, 0.244140625, 0.146484375,
#      0.146484375, 0.09765625, 0.09765625, 0.0003814697265625, 0.02288818359375,
#      0.0457763671875, 0.091552734375, 0.0003814697265625]

# 除以4
# data_transmission_t = \
#     [1.675605775, 15.283203125, 7.6416015625, 7.6416015625, 3.8330078125,
#      3.8330078125, 1.9287109375, 1.9287109375, 0.9765625, 0.9765625, 0.48828125,
#      0.48828125, 0.244140625, 0.244140625, 0.1220703125, 0.1220703125, 0.0732421875,
#      0.0732421875, 0.048828125, 0.048828125, 0.00019073486328125, 0.011444091796875,
#      0.02288818359375, 0.0457763671875, 0.00019073486328125]

data_transmission_t = \
    [6.70242310e+00, 6.11328125e+01, 3.05664062e+01, 3.05664062e+01, 1.53320312e+01,
     1.53320312e+01, 7.71484375e+00, 7.71484375e+00, 3.90625000e+00, 3.90625000e+00,
     1.95312500e+00, 1.95312500e+00, 9.76562500e-01, 9.76562500e-01, 4.88281250e-01,
     4.88281250e-01, 2.92968750e-01, 2.92968750e-01, 1.95312500e-01, 1.95312500e-01,
     7.62939453e-04, 4.57763672e-02, 9.15527344e-02, 1.83105469e-01, 7.62939453e-04]

# 乘以2
# data_transmission_t = \
#     [13.4048462, 122.265625, 61.1328124, 61.1328124, 30.6640624,
#      30.6640624, 15.4296875, 15.4296875, 7.8125, 7.8125, 3.90625,
#      3.90625, 1.953125, 1.953125, 0.9765625, 0.9765625, 0.5859375,
#      0.5859375, 0.390625, 0.390625, 0.00152587890625, 0.0915527344,
#      0.183105469, 0.366210938, 0.00152587890625]

# 乘以4
# data_transmission_t = \
#     [26.8096924, 244.53125, 122.265625, 122.265625, 61.1328125,
#      61.1328125, 30.6640625, 30.6640625, 15.625, 15.625, 7.8125,
#      7.8125, 3.90625, 3.90625, 1.953125, 1.953125, 1.171875,
#      1.171875, 0.78125, 0.78125, 0.0030517578125, 0.18310546875,
#      0.3662109375, 0.732421875, 0.0030517578125]

# 随机噪声0.05
# data_transmission_t = \
#     [7.29359436e+00, 6.23559493e+01, 3.20622313e+01, 3.39912088e+01, 1.67637041e+01,
#      1.45828485e+01, 8.08133293e+00, 7.65645889e+00, 3.88609007e+00, 3.98644502e+00,
#      1.96719175e+00, 2.09514390e+00, 1.01372255e+00, 9.82503663e-01, 4.99117755e-01,
#      4.96427596e-01, 3.14854674e-01, 2.89963502e-01, 1.98369802e-01, 1.86971721e-01,
#      6.65550620e-04, 4.72723814e-02, 9.55098093e-02, 1.76310745e-01, 8.49523721e-04]

# 随机噪声0.05
# data_transmission_t = \
#     [7.29359436, 62.3559493, 32.0622313, 33.9912088, 16.7637041,
#      14.5828485, 8.08133293, 7.65645889, 3.88609007, 3.98644502,
#      1.96719175, 2.0951439, 1.01372255, 0.982503663, 0.499117755,
#      0.496427596, 0.314854674, 0.289963502, 0.198369802, 0.186971721,
#      0.00066555062, 0.0472723814, 0.0955098093, 0.176310745, 0.000849523721]

# 随机噪声0.1
# data_transmission_t = \
#     [7.03534194, 60.2875639, 32.5461573, 35.2217611, 14.9730265,
#      14.9730517, 8.93318176, 8.30690765, 3.72286157, 4.11818752,
#      1.86261373, 1.86216216, 1.00019163, 0.789718726, 0.404056746,
#      0.460825804, 0.263295963, 0.302175215, 0.177577655, 0.167728443,
#      0.00087475958, 0.0447428453, 0.0921709736, 0.157017551, 0.000721406347]

# # 随机噪声0.15
# data_transmission_t = \
#     [8.47593688e+00, 6.48022228e+01, 3.50538816e+01, 4.08408140e+01, 1.96270498e+01,
#      1.30844830e+01, 8.81431130e+00, 7.53968917e+00, 3.84577020e+00, 4.14683506e+00,
#      1.99532526e+00, 2.37918169e+00, 1.08804264e+00, 9.94385989e-01, 5.20790764e-01,
#      5.12720288e-01, 3.58626522e-01, 2.83953006e-01, 2.04484405e-01, 1.70290164e-01,
#      4.70772955e-04, 5.02644099e-02, 1.03423959e-01, 1.62721298e-01, 1.02269226e-03]

# # 随机噪声0.2
# data_transmission_t = \
#     [9.06710814, 66.0253596, 36.5497068, 44.2656166, 21.0587227,
#      12.3353002, 9.18080049, 7.48130431, 3.82561027, 4.22703008,
#      2.00939202, 2.52120059, 1.12520268, 1.00032715, 0.531627269,
#      0.520866634, 0.380512446, 0.280947758, 0.207541707, 0.161949385,
#      0.000373384122, 0.0517604242, 0.107381034, 0.155926574, 0.00110927652]

# 除以2
# data_transmission_t_ = \
#     [2.68096924, 24.53125, 12.265625, 12.265625, 6.1328125,
#      6.1328125, 3.0859375, 3.0859375, 1.5625, 1.5625, 0.78125,
#      0.78125, 0.390625, 0.390625, 0.1953125, 0.1953125, 0.1171875,
#      0.1171875, 0.078125, 0.078125, 0.00030517578125, 0.018310546875,
#      0.03662109375, 0.0732421875, 0.00030517578125]

# 除以4
# data_transmission_t_ = \
#     [1.34048462, 12.265625, 6.1328125, 6.1328125, 3.06640625,
#      3.06640625, 1.54296875, 1.54296875, 0.78125, 0.78125, 0.390625,
#      0.390625, 0.1953125, 0.1953125, 0.09765625, 0.09765625, 0.05859375,
#      0.05859375, 0.0390625, 0.0390625, 0.000152587890625, 0.0091552734375,
#      0.018310546875, 0.03662109375, 0.000152587890625]

# data_t_
data_transmission_t_ = \
    [5.36193848e+00, 4.89062500e+01, 2.44531250e+01, 2.44531250e+01,
     1.22656250e+01, 1.22656250e+01, 6.17187500e+00, 6.17187500e+00,
     3.12500000e+00, 3.12500000e+00, 1.56250000e+00, 1.56250000e+00,
     7.81250000e-01, 7.81250000e-01, 3.90625000e-01, 3.90625000e-01,
     2.34375000e-01, 2.34375000e-01, 1.56250000e-01, 1.56250000e-01,
     6.10351562e-04, 3.66210938e-02, 7.32421875e-02, 1.46484375e-01,
     6.10351562e-04]

# 乘以2
# data_transmission_t_ = \
#     [10.72387696, 97.8125, 48.90625, 48.90625, 24.53125,
#      24.53125, 12.34375, 12.34375, 6.25, 6.25, 3.125,
#      3.125, 1.5625, 1.5625, 0.78125, 0.78125, 0.46875,
#      0.46875, 0.3125, 0.3125, 0.001220703125, 0.0732421875,
#      0.146484375, 0.29296875, 0.001220703125]

# 乘以4
# data_transmission_t_ = \
#     [42.89550784, 391.25, 195.625, 195.625, 98.125,
#      98.125, 49.375, 49.375, 25.0, 25.0, 12.5,
#      12.5, 6.25, 6.25, 3.125, 3.125, 1.875,
#      1.875, 1.25, 1.25, 0.0048828125, 0.29296875,
#      0.5859375, 1.171875, 0.0048828125]

# 随机噪声0.05
# data_transmission_t_ = \
#     [4.97202752e+00, 4.90181439e+01, 2.42242635e+01, 2.63271871e+01, 1.31667552e+01,
#      1.23606514e+01, 6.28857359e+00, 5.89790987e+00, 2.81550055e+00, 3.07063873e+00,
#      1.57471476e+00, 1.65861646e+00, 8.28217963e-01, 7.66120046e-01, 3.84720649e-01,
#      3.70145450e-01, 2.17734165e-01, 2.14379646e-01, 1.71490433e-01, 1.52268342e-01,
#      5.96982595e-04, 3.43271570e-02, 7.60894422e-02, 1.34663834e-01, 6.03859244e-04]

# 随机噪声0.05
# data_transmission_t_ = \
#     [4.97202752, 49.0181439, 24.2242635, 26.3271871, 13.1667552,
#      12.3606514, 6.28857359, 5.89790987, 2.81550055, 3.07063873,
#      1.57471476, 1.65861646, 0.828217963, 0.766120046, 0.384720649,
#      0.37014545, 0.217734165, 0.214379646, 0.171490433, 0.152268342,
#      0.000596982595, 0.034327157, 0.0760894422, 0.134663834, 0.000603859244]

# # 随机噪声0.1
# data_transmission_t_ = \
#     [5.42141449, 43.277172, 25.3718241, 22.9843757, 11.9078444,
#      11.5275942, 7.31507794, 6.16354468, 2.79446533, 3.38204529,
#      1.37174318, 1.59513494, 0.628150772, 0.677485465, 0.398314892,
#      0.419471351, 0.238391444, 0.231664493, 0.151545255, 0.133148094,
#      0.000566415758, 0.0349341842, 0.0809847819, 0.151517846, 0.000502744131]

# 随机噪声0.15
# data_transmission_t_ = \
#     [4.19220559e+00, 4.92419316e+01, 2.37665405e+01, 3.00753113e+01, 1.49690155e+01,
#      1.25507041e+01, 6.52197077e+00, 5.34997960e+00, 2.19650166e+00, 2.96191618e+00,
#      1.59914429e+00, 1.85084938e+00, 9.22153889e-01, 7.35860139e-01, 3.72911948e-01,
#      3.29186350e-01, 1.84452494e-01, 1.74388939e-01, 2.01971298e-01, 1.44305027e-01,
#      5.70244662e-04, 2.97392833e-02, 8.17839517e-02, 1.11022752e-01, 5.90874608e-04]

# 随机噪声0.2
# data_transmission_t_ = \
#     [3.80229463, 49.3538255, 23.537679, 31.9493733, 15.8701457,
#      12.6457304, 6.63866936, 5.07601447, 1.88700221, 2.90755491,
#      1.61135905, 1.94696584, 0.969121851, 0.720730185, 0.367007598,
#      0.3087068, 0.167811659, 0.154393585, 0.217211731, 0.140323369,
#      0.000556875695, 0.0274453465, 0.0846312064, 0.0992022115, 0.00058438229]
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
    def __init__(self, alpha=0.5, gamma=0.95, epsilon=0.1, learning_rate=0.1):
        super(TaskOffloadingEnv, self).__init__()
        self.alpha = alpha
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate
        self.learning_rate = learning_rate  # Learning rate for Q-learning

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

        self.q_table = np.zeros((25, 2))  # Initialize Q-table with zero (25 states * 2 actions each)

        self.reset()

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
        return np.array(initial_state, dtype=np.float32)

    # def critic_evaluate(self):
    #     if not self.actions_taken:
    #         return 0
    #     critic_recommendations = partition_recommendations(2, latency_edge, latency_server, data_transmission_t, data_transmission_t_)
    #     print("Critic's partition recommendation:", critic_recommendations)
    #     matched_decisions = sum([1 if act == rec else 0 for act, rec in zip(self.actions_taken, critic_recommendations)])
    #     critic_reward = matched_decisions / len(self.actions_taken)
    #     return critic_reward

    def step(self, action):
        if self.current_task == 25:
            # All tasks are done
            return self.reset(), 0, True, {}

        # Add the current action to the actions_taken list
        self.actions_taken.append(action)

        if np.random.random() < self.epsilon:
            action = self.action_space.sample()  # Exploration: random action
        else:
            action = np.argmax(self.q_table[self.current_task])  # Exploitation: best known action

        if action == 0:  # Execute on edge
            latency = latency_edge[self.current_task] + data_transmission_t[self.current_task]
            # throughput = throughput_edge[self.current_task]
        else:  # Execute on server
            latency = latency_server[self.current_task] + data_transmission_t_[self.current_task]
            # throughput = throughput_server[self.current_task]

        # Estimate disturbance
        disturbance = self.estimate_disturbance(action)

        # Compute reward
        reward = self.calculate_reward(latency, disturbance)

        # Q-learning update
        self.q_table_update(action, reward)

        # Q-learning update
        old_value = self.q_table[self.current_task, action]
        next_max = np.max(self.q_table[(self.current_task + 1) % 25])
        self.q_table[self.current_task, action] = old_value + self.learning_rate * (
                    reward + self.gamma * next_max - old_value)

        # self.state.append(action[0])
        # Add the critic's evaluation to the reward
        # critic_reward = self.critic_evaluate()
        # print("Critic's reward:", critic_reward)
        # normalized_critic_reward = (critic_reward - 0) / (1 - 0)  # Since min is 0 and max is 1
        # reward += critic_weight * normalized_critic_reward
        print("Reward after adding critic's reward:", reward)
        self.current_task += 1

        if self.current_task == 25:
            done = True
        else:
            done = False

        next_state = self.reset() if done else self.get_state()

        return np.array(next_state), reward, done, {}

    def estimate_disturbance(self, action):
        # Placeholder for a real disturbance estimation based on the difference in expected and actual latency
        expected_latency = latency_edge if action == 0 else latency_server
        actual_latency = np.random.normal(loc=expected_latency, scale=0.1)  # Simulate actual with noise
        return actual_latency - expected_latency

    def calculate_reward(self, latency, disturbance):
        # Calculate throughput for comparison; using max as a placeholder to simulate operational conditions
        throughput = max(
            sum(latency_edge[:self.current_task + 1]) + data_transmission_t[self.current_task],
            sum(latency_server[self.current_task:]) + data_transmission_t_[self.current_task])

        # Normalize the latency and throughput to the range [0, 1] for consistent reward scaling
        normalized_latency = (latency - min(latency_edge.min(), latency_server.min())) / \
                             (max(latency_edge.max(), latency_server.max()) - min(latency_edge.min(),
                                                                                            latency_server.min()))
        normalized_throughput = (throughput - min(latency_edge.min(), latency_server.min())) / \
                                (max(latency_edge.max(), latency_server.max()) - min(latency_edge.min(),
                                                                                               latency_server.min()))

        # Calculate the disturbance impact
        disturbance_impact = np.linalg.norm(disturbance)  # Use norm for vector disturbance or absolute value for scalar

        # Calculate reward considering both latency and throughput, penalize based on disturbance
        # Alpha controls the balance between latency and throughput importance
        reward = -self.alpha * np.log(normalized_latency + 1e-3) + \
                 (1 - self.alpha) * np.log(normalized_throughput + 1e-3) - disturbance_impact

        return reward

    def q_table_update(self, action, reward):
        old_value = self.q_table[self.current_task, action]
        next_max = np.max(self.q_table[(self.current_task + 1) % 25])
        self.q_table[self.current_task, action] = old_value + self.learning_rate * (
                    reward + self.gamma * next_max - old_value)

    def get_state(self):
        return [self.compute_requirements[self.current_task], self.data_transmission_requirements[self.current_task],
                self.current_task, 25 - self.current_task, 0, self.previous_action]

    def render(self, mode="human"):
        pass


if __name__ == "__main__":
    env = TaskOffloadingEnv(alpha=0.7)
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()  # Randomly sample an action
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print("Total Reward after completing all tasks:", total_reward)
