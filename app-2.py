# 安装所需的库
# !pip install sklearn
# !pip install gensim

# 导入必要的库
# from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# 示例文本数据
texts = [
    "I love machine learning.",
    "Artificial intelligence is fascinating.",
    "I enjoy learning about new technologies.",
    "My favorite sport is football.",
    "I like playing tennis during weekends.",
    "Basketball is a great sport."
]

# 示例嵌入数据
embeddings = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.1, 0.4, 0.1],
    [0.9, 0.8, 0.7],
    [0.7, 0.6, 0.8],
    [0.8, 0.7, 0.6]
])

# 标准化嵌入数据
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# 使用KMeans聚类
num_clusters = 2  # 设置聚类数
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings_scaled)

# 获取聚类结果
labels = kmeans.labels_

# 输出每个文本的聚类标签
for text, label in zip(texts, labels):
    print(f"Text: {text} -> Cluster: {label}")