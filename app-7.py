## code search

import pandas as pd
import numpy as np

# 示例数据集
data = {
    'code': [
        'def add(a, b):\n    return a + b\n\n# Adds two numbers',
        'def subtract(a, b):\n    return a - b\n\n# Subtracts second number from first',
        'def multiply(a, b):\n    return a * b\n\n# Multiplies two numbers'
    ]
}
df = pd.DataFrame(data)

# 生成mock嵌入向量
def get_mock_embedding(text, embedding_dim=512):
    np.random.seed(hash(text) % (2**32 - 1))  # 为了使每个文本的嵌入是确定的
    return np.random.rand(embedding_dim)

# 计算余弦相似度
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

df['code_embedding'] = df['code'].apply(lambda x: get_mock_embedding(x))

def search_functions(df, code_query, n=3, n_lines=7):
    # 获取查询代码的嵌入
    query_embedding = get_mock_embedding(code_query)
    
    # 计算相似度
    df['similarities'] = df['code_embedding'].apply(lambda x: cosine_similarity(x, query_embedding))
    
    # 获取最相似的n个代码段
    result = df.sort_values('similarities', ascending=False).head(n)
    
    # 打印结果
    for idx, row in result.iterrows():
        print(f"Similarity: {row['similarities']:.4f}")
        print("\n".join(row['code'].split('\n')[:n_lines]))
        print("-" * 80)
    
    return result

# 运行代码搜索
search_query = 'Completions API tests'
search_results = search_functions(df, search_query, n=3)

# 打印搜索结果
for i, row in search_results.iterrows():
    print(f"Code {i+1}:\n{row['code']}\n")
