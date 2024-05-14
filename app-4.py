
## 分类.(但是不保证能用啊.)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 示例数据集
data = [
    ("I love this movie, it was fantastic!", "positive"),
    ("This film was terrible and boring.", "negative"),
    ("The plot was good but the characters were uninteresting.", "neutral"),
    ("An amazing experience, would definitely recommend!", "positive"),
    ("Not my cup of tea, won't watch again.", "negative"),
]

# 准备数据
texts, labels = zip(*data)

# 生成随机嵌入向量
def get_mock_embeddings(texts, embedding_dim=512):
    return np.random.rand(len(texts), embedding_dim)

embeddings = get_mock_embeddings(texts)

# 标签编码
label_to_int = {"positive": 0, "negative": 1, "neutral": 2}
int_to_label = {v: k for k, v in label_to_int.items()}
y = np.array([label_to_int[label] for label in labels])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)

# 训练分类器
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# 预测
y_pred = classifier.predict(X_test)

# 结果报告
labels = list(label_to_int.values())
target_names = list(label_to_int.keys())
print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))
