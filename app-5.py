## 情感分类.

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 示例文本和标签
reviews = ["hello", "helloya", "i am newnew"]
scores = [1, 5, 3]  # 示例分数：1-负面，5-正面，3-中性

# 创建一个数据框来存储示例数据
import pandas as pd
df = pd.DataFrame({'Review': reviews, 'Score': scores})

# 过滤掉中性的评分（3分）
df = df[df.Score != 3]

# 将评分转换为情感标签
df['sentiment'] = df.Score.replace({1: 'negative', 2: 'negative', 4: 'positive', 5: 'positive'})

# 定义情感标签
labels = ['negative', 'positive']

# 模拟生成嵌入向量的函数
def get_mock_embedding(text, embedding_dim=512):
    np.random.seed(hash(text) % (2**32))
    return np.random.rand(embedding_dim)

# 生成情感标签的嵌入向量
label_embeddings = [get_mock_embedding(label) for label in labels]

# 定义计算标签得分的函数
def label_score(review, label_embeddings):
    review_embedding = get_mock_embedding(review)
    positive_similarity = cosine_similarity([review_embedding], [label_embeddings[1]])[0][0]
    negative_similarity = cosine_similarity([review_embedding], [label_embeddings[0]])[0][0]
    return positive_similarity - negative_similarity

# 对示例文本进行预测
for review in df['Review']:
    score = label_score(review, label_embeddings)
    prediction = 'positive' if score > 0 else 'negative'
    print(f"Review: '{review}' is predicted as {prediction}")
