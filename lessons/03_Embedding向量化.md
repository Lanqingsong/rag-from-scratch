# 03 · Embedding 向量化

> 对应代码：`4.向量和Embeddings.py`

---

## 什么是 Embedding？

Embedding（嵌入）是把文本转换成**一组数字（向量）**的过程。  
这组数字编码了文本的语义信息，语义相近的文本，对应的向量在空间中距离也近。

```
"镜头焦距如何计算"    → [0.12, -0.87, 0.45, 0.33, ...]  (3072 个数字)
"f 值怎么求"          → [0.11, -0.84, 0.47, 0.31, ...]  (距离很近 ✓)
"今天天气真好"         → [-0.63, 0.21, -0.55, 0.78, ...] (距离很远 ✓)
```

---

## 为什么用向量而不是关键词匹配？

**关键词搜索：**
- 搜索"焦距"，找不到包含"f 值"的文档（字面不同）
- 搜索"工作距离"，找不到只写"WD"的文档

**向量搜索：**
- 模型学会了这些词在语义上是相关的
- "WD"和"工作距离"的向量相近，可以被检索到

---

## 高维空间的直觉理解

千维向量难以想象，但在低维空间里，逻辑完全相同：

```
                  "镜头焦距" ●
                              ↑ 语义相近
                  "f值计算" ●

                                      ● "今天天气" ← 语义无关，距离远
```

**余弦相似度**是衡量两个向量"方向相似程度"的指标：

```
cos(θ) = (A · B) / (|A| × |B|)

结果范围：-1（完全相反）到 1（完全相同）
```

本项目的相关度阈值设为 0.62，低于此值的结果直接丢弃。

---

## 核心代码解读

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def get_embeddings(texts, model="text-embedding-v3"):
    data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]

vec = get_embeddings(["我爱你", "你好"])
print(len(vec))     # 2（两个文本，两个向量）
print(len(vec[0]))  # 3072（每个向量的维度）
```

**关键点：**
- `input` 接受一个列表，批量处理更高效
- 返回的每个向量是一个 float 列表，长度即维度
- 向量维度由模型决定：`text-embedding-v2` 是 1536 维，`text-embedding-v3` 是 3072 维

---

## 模型选择

| 模型 | 维度 | 特点 |
|------|------|------|
| `text-embedding-v2`（千问） | 1536 | 本项目使用，中文优化 |
| `text-embedding-v3`（千问） | 3072 | 更高精度，成本稍高 |
| `text-embedding-ada-002`（OpenAI） | 1536 | 英文优秀，中文一般 |

**为什么选千问 Embedding？**  
通用英文 Embedding 模型对中文专业术语（如"工业相机"、"图像传感器"）的语义理解较差，  
使用中文优化的模型能显著提升检索精度。

---

## 批处理与 API 限制

大量文档入库时需要批量向量化，但 API 单次调用有文本数量限制（千问约 25 条/次）：

```python
# 主项目 knowledge_base.py 中的批处理实现
def embed_documents(self, texts):
    all_embeddings = []
    for i in range(0, len(texts), self.batch_size):  # batch_size = 25
        batch = texts[i:i + self.batch_size]
        response = TextEmbedding.call(model=Config.EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([r['embedding'] for r in response.output['embeddings']])
    return all_embeddings
```

---

## 动手练习

1. 运行 `4.向量和Embeddings.py`，查看向量维度
2. 计算"我爱你"和"I love you"的向量，用 `numpy` 计算余弦相似度
3. 计算"镜头焦距"和"f值计算"的相似度，再和"今天天气"对比

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

vecs = get_embeddings(["镜头焦距", "f值计算", "今天天气"])
print(cosine_similarity(vecs[0], vecs[1]))  # 期望 > 0.8
print(cosine_similarity(vecs[0], vecs[2]))  # 期望 < 0.5
```

---

## 主项目中的对应代码

`knowledge_base.py` 中的 `QianwenEmbeddings` 类实现了 LangChain 的 `Embeddings` 接口，  
FAISS 向量库在 `build_vector_store()` 时调用它批量向量化所有 chunk，  
检索时对查询语句也调用同一个 `embed_query()` 方法。
