## 聚类
import numpy as np
from sklearn.cluster import KMeans

# 示例嵌入数据 (每个字符串的嵌入向量)
embeddings = {
    "hello": np.random.rand(1, 5),       # 随机生成的嵌入向量
    "helloya": np.random.rand(1, 5),     # 随机生成的嵌入向量
    "i am newnew": np.random.rand(1, 5)  # 随机生成的嵌入向量
}

# 将嵌入向量转换为矩阵
embedding_matrix = np.vstack(list(embeddings.values()))

# 设置聚类数量
num_clusters = 3

# 使用 KMeans 进行聚类
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
kmeans.fit(embedding_matrix)

# 将聚类标签添加到数据中
clusters = kmeans.labels_

# 打印结果
for i, key in enumerate(embeddings.keys()):
    print(f"'{key}' 被分配到集群 {clusters[i]}")
