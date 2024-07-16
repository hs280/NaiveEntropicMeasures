import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Generate some random data
data = np.random.rand(10, 12)

# Create the heatmap with seaborn
ax = sns.heatmap(data, cmap='YlGnBu', linewidths=1, linecolor='white')

# Manually draw lines between cells using annotate
for i in range(data.shape[0]):
    ax.annotate('', xy=(-0.5, i), xytext=(data.shape[1], i),
                xycoords='data', textcoords='data',
                arrowprops=dict(color='blue', linestyle='--', linewidth=1, arrowstyle='-'),
                annotation_clip=False)

for j in range(data.shape[1]):
    ax.annotate('', xy=(j + 1, 0), xytext=(j + 1, data.shape[0] + 0.5),
                xycoords='data', textcoords='data',
                arrowprops=dict(color='grey', linestyle='--', linewidth=1, arrowstyle='-'),
                annotation_clip=False)

# Annotate the plot
plt.title('Heatmap with Manual Grid Lines')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

plt.show()
