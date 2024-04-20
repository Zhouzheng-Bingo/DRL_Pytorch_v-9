# Complete code from loading the CSV to plotting the final graph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the CSV file into a DataFrame
focused_data_df = pd.read_csv('./3Ddata/无critic_dqn立体行为分布.csv')

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Prepare data for plotting
# We will plot batches as 'Episodes' on the x-axis, and each point within a batch as 'Task Index' on the y-axis
episodes = np.arange(1, len(focused_data_df.columns) + 1)  # 1 to number of batches
task_indices = np.arange(1, 26)  # 1 to 25, since each batch has 25 points

# Create mesh grids for the episodes and task indices
X, Y = np.meshgrid(episodes, task_indices)
Z_edge_node = np.zeros_like(X, dtype=float)
Z_edge_server = np.zeros_like(X, dtype=float)

# Populate Z arrays for Edge Node and Edge Server
for i, batch in enumerate(focused_data_df.columns):
    Z_edge_node[:, i] = np.where(focused_data_df[batch] == 0, 0, np.nan)  # Edge Node has Offload Ratio 0
    Z_edge_server[:, i] = np.where(focused_data_df[batch] == 1, 1, np.nan)  # Edge Server has Offload Ratio 1

# Plot Edge Node and Edge Server
ax.scatter(X, Y, Z_edge_node, c='r', marker='o', label='Edge Node')
ax.scatter(X, Y, Z_edge_server, c='b', marker='^', label='Edge Server')

# Set labels and legend
ax.set_xlabel('Episode')
ax.set_ylabel('Task Index')
ax.set_zlabel('Offload Location')

# Setting the axis ticks based on the provided specifications
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_yticks([0, 5, 10, 15, 20, 25])
ax.set_zticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Setting the axis tick labels
ax.set_xticklabels([0, 2, 4, 6, 8])
ax.set_yticklabels([0, 5, 10, 15, 20, 25])
ax.set_zticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Add a legend to the plot
ax.legend()

# Show the plot
plt.show()
