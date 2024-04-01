import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Ensure the font is set to Times New Roman and bold for all text elements
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'

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

# Convert all latencies to milliseconds for consistency in display
latency_edge_ms = [x * 1000 for x in latency_edge]
latency_server_ms = [x * 1000 for x in latency_server]
data_transmission_t_ms = [x for x in data_transmission_t]
data_transmission_t__ms = [x for x in data_transmission_t_]

# Combine all the data into a 2D array where each row represents a different measurement type
data = np.array([latency_edge_ms, latency_server_ms, data_transmission_t_ms, data_transmission_t__ms])

# Use 'imshow' function to create a heatmap with 'tab20c_r' colormap
plt.figure(figsize=(10, 8))
plt.imshow(data, aspect='auto', cmap=matplotlib.colormaps['tab20c'])

# Add color bar on the right
cbar = plt.colorbar()
cbar.set_label('Delay (ms)', rotation=270, labelpad=15)

# Add titles and labels
plt.title('Heatmap of Delays', fontsize=16, fontweight='bold')
plt.xlabel('Subtask')

# Define the y-axis labels vertically
measurement_types = ['Edge Node Execution', 'Edge Server Execution', 'Edge Node Upload', 'Edge Server Download']
plt.yticks(np.arange(len(measurement_types)), measurement_types, rotation=90, ha='right', va='center')

# Define the x-axis labels (horizontal)
plt.xticks(np.arange(len(latency_edge_ms)), [f"{i+1}" for i in range(25)])

# Show the plot
plt.tight_layout()
plt.show()
