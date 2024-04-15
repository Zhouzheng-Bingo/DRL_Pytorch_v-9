import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Setting the font to Times New Roman
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')

# Load the CSV data into a DataFrame
# df = pd.read_csv('./3Ddata/offload_data.csv')
df = pd.read_csv('./3Ddata/offload_data_ppo.csv')


# Extract columns from the DataFrame
episodes = df['Episode'].values
task_index = df['Task_Index'].values
offload_location = df['Action'].values

# Creating a new figure for 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for 'edge node' offload location
local_mask = offload_location == 0
ax.scatter(episodes[local_mask], task_index[local_mask], offload_location[local_mask], c='r', marker='o', label='Edge Node')

# Scatter plot for 'edge server' offload location
server_mask = offload_location == 1
ax.scatter(episodes[server_mask], task_index[server_mask], offload_location[server_mask], c='b', marker='^', label='Edge Server')

# Labeling the axes
ax.set_xlabel('Episode')
ax.set_ylabel('Task Index')
ax.set_zlabel('Offload Location')

# Adding a title and a legend
# ax.set_title('(a)')
ax.legend()

# Show the plot
plt.show()
