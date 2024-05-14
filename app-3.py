### 异常检测.

# !pip install scikit-learn

import numpy as np
from sklearn.neighbors import NearestNeighbors

# 定义一个函数来获取文本的嵌入表示
# def get_embedding(text):
#     response = openai.Embedding.create(input=text, engine="text-embedding-ada-002")
#     return response['data'][0]['embedding']


# 定义一个生成mock数据的get_embedding函数
def get_embedding(text):
    # 模拟嵌入数据，假设每个嵌入向量有512个维度
    np.random.seed(hash(text) % 2**32)  # 确保每个文本生成的嵌入是可重复的
    return np.random.rand(512)

# 生成一些示例数据
normal_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaps over a sleepy dog.",
    "Speedy foxes jump over lazy dogs often.",
    "A quick brown fox jumps over a lazy dog."
]

anomalous_texts = [
    "Suddenly, a spaceship appeared in the sky.",
    "The stock market crashed overnight.",
    "A volcano erupted unexpectedly."
]

# 获取正常数据的嵌入
normal_embeddings = np.array([get_embedding(text) for text in normal_texts])

# 获取异常数据的嵌入
anomalous_embeddings = np.array([get_embedding(text) for text in anomalous_texts])

# 使用k-最近邻算法来进行异常检测
nbrs = NearestNeighbors(n_neighbors=2).fit(normal_embeddings)

# 获取正常数据的邻居距离
distances, _ = nbrs.kneighbors(normal_embeddings)
mean_distance = np.mean(distances[:, 1])

# 定义一个阈值（这里使用平均距离的两倍作为阈值）
threshold = 2 * mean_distance

# 检测异常数据
for text, embedding in zip(anomalous_texts, anomalous_embeddings):
    distance, _ = nbrs.kneighbors([embedding])
    if distance[0, 1] > threshold:
        print(f"Anomalous text detected: {text} with distance {distance[0, 1]}")
    else:
        print(f"Normal text: {text} with distance {distance[0, 1]}")
