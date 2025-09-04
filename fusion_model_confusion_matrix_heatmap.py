import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Average confusion matrix (as provided)
conf_matrix = np.array([
    [5.0, 3.8, 6.0],
    [2.4, 6.8, 9.8],
    [0.8, 4.4, 40.6]
])

labels = ['Unpleasantness', 'Neutral', 'Pleasantness']

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=labels, yticklabels=labels, cbar=True)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Average Confusion Matrix of Fusion Model')
plt.tight_layout()
plt.show()
