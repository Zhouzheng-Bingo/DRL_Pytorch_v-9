import psutil
import pandas as pd
import matplotlib.pyplot as plt
import time
import GPUtil


def monitor_system(interval=1, duration=60):
    # 初始化数据存储列表
    times = []
    cpu_usages = []
    memory_usages = []
    gpu_usages = []
    gpu_mem_usages = []

    start_time = time.time()
    while (time.time() - start_time) < duration:
        # CPU和内存使用率
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().percent
        cpu_usages.append(cpu_usage)
        memory_usages.append(memory_usage)
        times.append(time.time() - start_time)

        # GPU使用率和内存使用率
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_usage = gpus[0].load * 100  # 转换为百分比
            gpu_memory_usage = gpus[0].memoryUsed / gpus[0].memoryTotal * 100
            gpu_usages.append(gpu_usage)
            gpu_mem_usages.append(gpu_memory_usage)
        else:
            gpu_usages.append(0)
            gpu_mem_usages.append(0)

        time.sleep(interval)

    # 将数据保存到DataFrame
    data = {
        "Time": times,
        "CPU Usage (%)": cpu_usages,
        "Memory Usage (%)": memory_usages,
        "GPU Usage (%)": gpu_usages,
        "GPU Memory Usage (%)": gpu_mem_usages
    }
    df = pd.DataFrame(data)
    df.to_csv("system_performance.csv", index=False)

    # 绘图
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(df['Time'], df['CPU Usage (%)'], label="CPU Usage (%)")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(df['Time'], df['Memory Usage (%)'], label="Memory Usage (%)")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(df['Time'], df['GPU Usage (%)'], label="GPU Usage (%)")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(df['Time'], df['GPU Memory Usage (%)'], label="GPU Memory Usage (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("performance_plots.png")
    plt.show()


if __name__ == '__main__':
    monitor_system(interval=1, duration=10)  # 每1秒采样一次，持续60秒
