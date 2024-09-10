import numpy as np
import matplotlib.pyplot as plt

# Confusion matrix
conf_matrix = np.array([[91, 4, 1, 2, 2, 0, 0], 
                        [3, 89, 3, 2, 3, 0, 0],
                        [2, 2, 90, 3, 3, 0, 0],
                        [2, 1, 3, 88, 6, 0, 0],
                        [1, 2, 2, 4, 91, 0, 0],
                        [0, 0, 0, 0, 0, 100, 0],
                        [0, 0, 0, 0, 0, 0, 100]])

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, cmap='Blues')

# Add color bar
plt.colorbar()

# Add annotations
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black')

# Add labels
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Add class labels
class_labels = ['Grass', 'Field', 'Industry', 'River Lake', 'Forest', 'Resident', 'Parking']

plt.xticks(np.arange(len(class_labels)), class_labels, rotation=45)
plt.yticks(np.arange(len(class_labels)), class_labels)

plt.tight_layout()
plt.show()