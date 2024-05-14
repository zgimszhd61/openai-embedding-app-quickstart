# !pip install openai
import os
import openai

# 设置 API 密钥
os.environ["OPENAI_API_KEY"] = "sk-proj-"

# 示例文本
texts = [
    "This is a test sentence.",
    "Another example of a sentence.",
    "Text embeddings are useful for various NLP tasks."
]

# 获取嵌入
response = openai.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

# 打印响应
# print(response)

# 提取嵌入向量
mlist = response.data
embeddings = []
for item in mlist:
  embeddings = embeddings + [item.embedding]
  print(item.embedding)

# 查询文本
query_text = "Find similar sentences."
query_response = openai.embeddings.create(
    model="text-embedding-3-small",
    input=[query_text]
)

# print(query_response.data[0].embedding)

query_embedding = query_response.data[0].embedding
# print(query_embedding)

import numpy as np


def cosine_similarity(vector1, vector2):
    """
    计算两个向量之间的余弦相似度
    """
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    return dot_product / (norm_vector1 * norm_vector2)


def find_most_similar_text(embeddings, query_embedding):
    """找到与查询嵌入最相似的文本"""
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    most_similar_index = np.argmax(similarities)
    return most_similar_index, similarities[most_similar_index]

# 计算余弦相似度并找到最相似的文本
most_similar_index, similarity_score = find_most_similar_text(embeddings, query_embedding)

print(f"Most similar text: {texts[most_similar_index]}")
print(f"Similarity score: {similarity_score}")
