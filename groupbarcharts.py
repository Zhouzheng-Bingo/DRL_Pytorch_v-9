import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == '__main__':

    # 数据定义
    latency_edge_ms = [x * 1000 for x in latency_edge]
    latency_server_ms = [x * 1000 for x in latency_server]
    data_transmission_t_ms = data_transmission_t
    data_transmission_t__ms = data_transmission_t_

    # 定义子任务
    subtasks = [f"{i + 1}" for i in range(25)]

    # 设置图表大小
    plt.figure(figsize=(15, 8))

    # 设置每组数据的位置
    barWidth = 0.2
    r1 = np.arange(len(latency_edge_ms))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    # 绘制柱状图
    plt.bar(r1, latency_edge_ms, color='blue', width=barWidth, edgecolor='gray', label='Edge node execution delay')
    plt.bar(r2, data_transmission_t_ms, color='orange', width=barWidth, edgecolor='gray', label='Edge node data upload')
    plt.bar(r3, latency_server_ms, color='green', width=barWidth, edgecolor='gray', label='Edge server execution delay')
    plt.bar(r4, data_transmission_t__ms, color='red', width=barWidth, edgecolor='gray', label='Edge server data download')

    # 添加标签和标题
    plt.xlabel('Subtask', fontweight='bold')
    plt.ylabel('Delay (ms)', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(latency_edge_ms))], subtasks)

    # 创建图例
    plt.legend()

    # 显示图表
    plt.show()
