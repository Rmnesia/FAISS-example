import numpy as np
import faiss

from text2vec import SentenceModel



model = SentenceModel('shibing624/text2vec-base-chinese')
dimension = 384
index = faiss.IndexFlatL2(dimension)  # 创建一个FAISS索引
data = []

while True:
    text = input("请输入任意文本：")
    sentences = [text]

    embeddings = model.encode(sentences)

    distances, indices = index.search(embeddings, k=3)  # 查询最近的向量
    print("距离和编号为：%s %s" %(distances, indices))
    if distances[0][0] < 100:
        print("最相关的三句话是：1.%s 2.%s 3.%s" %(data[indices[0][0]],data[indices[0][1]], data[indices[0][2]]))

    data.append(text)
    index.add(embeddings)

