import matplotlib.pyplot as plt
import numpy as np

# 假设您的相似度矩阵存储在similarity_matrix变量中
similarity_matrix = np.random.rand(20, 20)

fig, ax = plt.subplots()
im = ax.imshow(similarity_matrix)

# 设置横纵坐标
ax.set_xticks(np.arange(20))
ax.set_yticks(np.arange(20))
ax.set_xticklabels([f'{i+1}' for i in range(20)])
ax.set_yticklabels([f'{i+1}' for i in range(20)])
ax.xlabel('Client')
ax.ylabel('Client')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# 在每个单元格中显示数值
# for i in range(20):
#     for j in range(20):
#         text = ax.text(j, i, round(similarity_matrix[i, j], 2), ha="center", va="center", color="w")

ax.set_title("Similarity Matrix Heatmap")
fig.tight_layout()
plt.show()