import psutil
import pandas as pd
import matplotlib.pyplot as plt
import time

def find_process_by_name(name):
    "查找并返回第一个匹配的进程名称"
    for process in psutil.process_iter(['name', 'exe', 'cmdline']):
        if name.lower() in process.info['name'].lower() or \
           process.info['exe'] and name.lower() in process.info['exe'].lower() or \
           process.info['cmdline'] and name.lower() in ' '.join(process.info['cmdline']).lower():
            return process
    return None

def monitor_process(pid, interval=1, duration=60):
    process = psutil.Process(pid)
    start_time = time.time()
    times = []
    cpu_usages = []
    memory_usages = []

    while (time.time() - start_time) < duration:
        try:
            cpu_usage = process.cpu_percent(interval=interval)
            memory_usage = process.memory_percent()
            times.append(time.time() - start_time)
            cpu_usages.append(cpu_usage)
            memory_usages.append(memory_usage)
        except psutil.NoSuchProcess:
            break  # 进程可能在监控过程中被关闭

    data = {
        "Time": times,
        "CPU Usage (%)": cpu_usages,
        "Memory Usage (%)": memory_usages
    }
    df = pd.DataFrame(data)
    df.to_csv("process_performance.csv", index=False)

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(df['Time'], df['CPU Usage (%)'], label='CPU Usage (%)')
    plt.title('CPU Usage Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('CPU Usage (%)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df['Time'], df['Memory Usage (%)'], label='Memory Usage (%)')
    plt.title('Memory Usage Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Memory Usage (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig("process_performance_plots.png")
    plt.show()


if __name__ == '__main__':
    pycharm_process = find_process_by_name('PyCharm(9)')
    if pycharm_process:
        monitor_process(pycharm_process.pid)
    else:
        print("PyCharm process not found.")
