import matplotlib.pyplot as plt
import numpy as np

# 假设数据 - 这些数据将根据实验结果更新
methods = ['Actor-Critic', 'Q-Learning', 'Critic+DQN']
response_times_low = [0.2, 0.3, 0.15]  # 低负载下的平均响应时间
response_times_medium = [0.5, 0.7, 0.4]  # 中等负载下的平均响应时间
response_times_high = [1.0, 1.5, 0.8]  # 高负载下的平均响应时间

barWidth = 0.25

# Set position of bar on X axis
r1 = np.arange(len(response_times_low))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.figure(figsize=(10,6))
plt.bar(r1, response_times_low, color='#7f6d5f', width=barWidth, edgecolor='grey', label='低负载')
plt.bar(r2, response_times_medium, color='#557f2d', width=barWidth, edgecolor='grey', label='中等负载')
plt.bar(r3, response_times_high, color='#2d7f5e', width=barWidth, edgecolor='grey', label='高负载')

# Add xticks on the middle of the group bars
plt.xlabel('方法', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(response_times_low))], methods)

# Create legend & Show graphic
plt.ylabel('平均响应时间 (秒)')
plt.title('不同负载条件下各算法的响应时间对比')
plt.legend()
plt.show()
