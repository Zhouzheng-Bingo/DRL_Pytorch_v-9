import os

import matplotlib

from TCN import TCN
from model import Net
import numpy as np
import matplotlib.pyplot as plt
import pywt
import torch
import time
from pulp import LpProblem, LpVariable, lpSum, LpMinimize
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('TkAgg')  # 使用TkAgg后端


# plt.style.use('seaborn-whitegrid')

# def partition(N, t, t_, data_t, data_t_):
#     latency_tolerable = 480
#
#     # Create a MILP problem with objective to minimize
#     model = LpProblem(name="optimal-cut", sense=LpMinimize)
#
#     # Define the binary decision variable y_i
#     y = [LpVariable(name=f"y_{i}", cat="Binary") for i in range(len(t))]
#
#     # Objective function to minimize total latency and maximize throughput
#     # This objective captures both minimizing latency and maximizing throughput
#     latency_difference = lpSum([abs((t[i] if i == 0 else sum(t[:i])) -
#                                     (t_[i] if i == len(t) - 1 else sum(t_[i + 1:]))) * y[i] for i in range(len(t))])
#     model += latency_difference
#
#     # Constraints to ensure the total latency is within the tolerable limits
#     for i in range(len(t)):
#         edge_latency = sum(t[:i]) + data_t[i]
#         cloud_latency = N * sum(t_[i:]) + N * data_t_[i]
#         model += edge_latency * y[i] + cloud_latency * (1 - y[i]) <= latency_tolerable
#
#     # Mock solution for testing since pulp is not available
#     best_throughout_point = 0
#     total_latency = 0
#     for i in range(len(t)):
#         if (sum(t[:i]) + data_t[i]) >= N * (sum(t_[i:]) + data_t_[i]):
#             best_throughout_point = i
#             total_latency = sum(t[:i]) + data_t[i] + N * (sum(t_[i:]) + data_t_[i])
#             break
#
#     latency_edge = [sum(t[:i]) for i in range(len(t))]
#     latency_server = [N * sum(t_[i:]) for i in range(len(t))]
#     data_transmission = [data_t[i] + N * data_t_[i] for i in range(len(t))]
#     throughput = [max((sum(t[:i]) + data_t[i]), N * (sum(t_[i:]) + data_t_[i])) for i in range(len(t))]
#     latency = [sum(t[:i]) + N * sum(t_[i:]) + data_t[i] + N * data_t_[i] for i in range(len(t))]
#     partition_point = 0
#     for i in range(len(t)):
#         if (sum(t[:i]) + N * sum(t_[i:]) + data_t[i] + N * data_t_[i]) <= latency_tolerable:
#             partition_point = i
#             break
#
#     return best_throughout_point, total_latency, latency_edge, latency_server, data_transmission, throughput, latency, partition_point

def partition(N, t, t_, data_t, data_t_):
    latency_tolerable = 480
    num_tasks = len(t)

    # 使用动态规划方法处理任务分配
    # dp[i] 表示前 i 个任务的最小延迟总和
    dp = [float('inf')] * (num_tasks + 1)
    dp[0] = 0  # 没有任务时的延迟为0

    # decision 记录决策，以便可以重建任务分配
    decision = [-1] * (num_tasks + 1)

    # 动态规划求解
    for i in range(1, num_tasks + 1):
        for j in range(i):
            edge_latency = sum(t[j:i]) + sum(data_t[j:i])
            cloud_latency = sum(t_[j:i]) + sum(data_t_[j:i])

            # 比较在边缘或云端执行任务 j 到 i-1 的总延迟
            current_latency = min(edge_latency, N * cloud_latency)
            if dp[j] + current_latency < dp[i]:
                dp[i] = dp[j] + current_latency
                decision[i] = j

    # 从 decision 重建任务分配方案
    i = num_tasks
    partition_point = -1
    while i > 0:
        if decision[i] != -1:
            partition_point = decision[i]  # 最后一个决策点为分配点
            break
        i -= 1

    # 重新计算各种指标以符合原函数输出
    best_throughout_point = partition_point
    total_latency = dp[num_tasks]
    latency_edge = [sum(t[:i + 1]) + sum(data_t[:i + 1]) for i in range(num_tasks)]
    latency_server = [sum(t_[i:]) + sum(data_t_[i:]) for i in range(num_tasks)]
    data_transmission = [data_t[i] + data_t_[i] for i in range(num_tasks)]
    throughput = [max(latency_edge[i], N * latency_server[i]) for i in range(num_tasks)]
    latency = [latency_edge[i] + latency_server[i] for i in range(num_tasks)]

    return best_throughout_point, total_latency, latency_edge, latency_server, data_transmission, throughput, latency, partition_point


def partition_recommendations(N, t, t_, data_t, data_t_):
    best_throughout_point, _, _, _, _, _, _, _ = partition(N, t, t_, data_t, data_t_)
    # 根据切分点为每个任务提供位置建议
    recommendations = ['0' if i <= best_throughout_point else '1' for i in range(len(t))]
    return recommendations


if __name__ == '__main__':
    tcn = TCN(1, 1, [4, 8, 16], kernel_size=3, dropout=0)
    res_net = Net(7, dropout=0)
    t0 = []
    for i in range(10):
        x = torch.rand((1, 7, 20000))
        t = time.time()
        x = x[:, :, ::4]
        A2, D2, D1 = pywt.wavedec(x, 'db4', mode='symmetric', level=2, axis=2)
        t0.append(time.time() - t)
    t0 = np.array(t0)
    t0 = list([np.mean(t0)])
    t1 = tcn.per_layer_time()
    tcn_total = np.array(t1).mean(0).sum()
    t1_ = tcn.per_layer_time(repeated=50)
    t1 = list(np.array(t1).mean(0))
    t1_ = list(np.array(t1_).mean(0))
    t2 = res_net.per_layer_time()
    t2 = list(np.array(t2).mean(0))
    t2_ = res_net.per_layer_time()
    t2_ = list(np.array(t2_).mean(0))

    data_size = [7 * 1255, 128 * 626, 128 * 313, 128 * 313, 128 * 157, 128 * 157, 128 * 79, 128 * 79, 128 * 40,
                 128 * 40, 128 * 20, 128 * 20,
                 128 * 10, 128 * 10, 128 * 5, 128 * 5, 128 * 3, 128 * 3, 128 * 2, 128 * 2, 1, 4 * 15, 8 * 15, 16 * 15,
                 1]
    data_t = np.array(data_size) * 4 * 8 * 1000 / (40 * 1024 * 1024)
    data_t_ = np.array(data_size) * 4 * 8 * 1000 / (50 * 1024 * 1024)
    t = np.array(t0 + t2 + t1)
    t = t / t.sum() * 380
    print(t[-4:].sum())
    t_ = np.array(t0 + t2_ + t1_)
    t_ = t_ / t_.sum() * 220
    print(t_[-4:].sum())

    fig, axes = plt.subplots(2, 1, dpi=200)

    axes[0].set_xticks(np.arange(0, 25, 2))
    axes[1].set_xticks(np.arange(0, 25, 2))
    '''
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    '''
    axes[0].grid(axis='x')
    axes[1].grid(axis='x')

    axes[0].bar(np.arange(len(t)), t, label='Edge device inference')
    axes[0].bar(np.arange(len(data_t)), data_t, bottom=t, label='Edge device data upload')
    axes[1].bar(np.arange(len(t_)), t_, label='Edge server inference')
    axes[1].bar(np.arange(len(data_t_)), data_t_, bottom=t_, label='Edge server data download')

    axes[0].set_ylabel('Latency(ms)')
    axes[0].set_ylim((0, 120))

    axes[1].set_ylabel('Latency(ms)')
    axes[1].set_xlabel('Subtasks')
    axes[1].set_ylim((0, 120))
    axes[0].legend()
    axes[1].legend()

    plt.savefig('.fig/ex6_fig1')

    best_throughout_point, total_latency, latency_edge, latency_server, data_transmission, throughput, latency, partition_point = partition(2, t, t_, data_t, data_t_)
    _, axs = plt.subplots(2, 1, dpi=200)
    for ax in axs:
        ax.set_xticks(np.arange(0, 25, 2))
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        ax.grid(axis='x')
    axs[0].bar(np.arange(len(latency_edge)), np.array(latency_edge), label='Edge device')
    axs[0].bar(np.arange(len(data_transmission)), np.array(data_transmission), bottom=latency_edge,
               label='Data transmission')
    axs[0].bar(np.arange(len(latency_server)), np.array(latency_server),
               bottom=np.array(latency_edge) + np.array(data_transmission), label='Edge server')
    axs[0].plot(np.array(throughput), 'k.-', label='Timeslice')
    axs[0].scatter(best_throughout_point, total_latency + 60, s=70, marker='d', color='r', label='Highest throughout')
    axs[0].scatter(partition_point, latency[partition_point] + 130, s=70, marker='*', color='gold',
                   label='Optimal partition point')
    axs[1].set_xlabel('Subtasks')
    axs[0].set_ylabel('Latency(ms)')
    axs[1].set_ylabel('Latency(ms)')
    axs[0].legend(bbox_to_anchor=(1, 1.3), ncol=3)
    best_throughout_point, total_latency, latency_edge, latency_server, data_transmission, throughput, latency, partition_point = partition(4, t, t_, data_t, data_t_)
    axs[1].bar(np.arange(len(latency_edge)), np.array(latency_edge), label='Edge device')
    axs[1].bar(np.arange(len(data_transmission)), np.array(data_transmission), bottom=latency_edge,
               label='Data transmission')
    axs[1].bar(np.arange(len(latency_server)), np.array(latency_server),
               bottom=np.array(latency_edge) + np.array(data_transmission), label='Edge server')
    axs[1].plot(np.array(throughput), 'k.-', label='Timeslice')
    axs[1].scatter(best_throughout_point, total_latency + 80, s=70, marker='d', color='r', label='Highest throughout')
    axs[1].scatter(partition_point, latency[partition_point] + 80, s=70, marker='*', color='gold',
                   label='Optimal partition point')
    plt.savefig('.fig/fig2')
