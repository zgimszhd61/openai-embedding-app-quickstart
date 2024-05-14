# !pip install sklearn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(0)
n_points = 20
X = np.random.normal(size=(n_points, 5))  # 生成20个点，每个点5维

# 设置 t-SNE 模型
tsne = TSNE(n_components=2, random_state=0, perplexity=5, n_iter=300)

# 运行 t-SNE 算法
X_embedded = tsne.fit_transform(X)

# 可视化结果
plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c='blue', marker='o')
plt.title('t-SNE Visualization of Mock Data')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.show()