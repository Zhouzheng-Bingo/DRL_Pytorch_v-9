import subprocess


# 温度暂时没用
def get_cpu_temp():
    result = subprocess.run(['vcgencmd', 'measure_temp'], stdout=subprocess.PIPE)
    return float(result.stdout.decode('utf-8').split('=')[1].split('\'')[0])

def get_cpu_usage():
    cmd = ["awk", "-v", "RS=''", "{print 1-($5*1.0)/($2+$3+$4+$5+$6+$7+$8)}", "/proc/stat"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    return float(result.stdout.decode('utf-8').strip())

def get_memory_usage():
    cmd = 'free | awk \'/内存/ {print $3/($3+$4+$6)}\''
    result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    return float(result.stdout.decode('utf-8').strip())

def get_gpu_power():
    return -1.0
