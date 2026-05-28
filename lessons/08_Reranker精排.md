# 08 · Reranker 精排

> 对应主项目 `knowledge_base.py` 中的 `QianwenReranker` 和 `search()` 的精排部分

---

## 为什么检索两步走？

向量检索（Embedding）是**召回阶段**：速度快，范围广，从万级文档中找出几十条候选。  
Reranker 是**精排阶段**：速度慢，精度高，对候选文档重新精确评分。

类似搜索引擎的两阶段架构：

```
海量文档库（万级）
    ↓ 粗检索（毫秒级，FAISS + BM25）
候选文档（十几条）
    ↓ 精排（百毫秒级，Reranker）
最终结果（3~5条）→ 送入 LLM
```

---

## 双编码器 vs 交叉编码器

这是理解 Reranker 的核心概念。

### 双编码器（Bi-encoder）= Embedding 模型的工作方式

```
文档  → [Encoder] → 向量A
查询  → [Encoder] → 向量B
相似度 = cosine(向量A, 向量B)
```

**特点：** 文档和查询**独立**编码，向量可以预计算存储。  
**缺点：** 编码时不知道对方，无法捕捉细粒度的交互信息。

### 交叉编码器（Cross-encoder）= Reranker 的工作方式

```
[查询 + 文档] → [Encoder] → 相关性分数
```

**特点：** 查询和文档**一起**输入模型，模型能充分看到两者的交互。  
**优点：** 精度远高于双编码器，能识别细微的语义差别。  
**缺点：** 无法预计算，每次检索都要实时跑一遍，速度慢。

---

## 为什么先召回再精排？

如果对全部文档都用 Reranker，速度会慢几十倍。工程上的解法：

```
10000 篇文档
    ↓ Embedding 快速粗筛（提前算好向量，查询时只算 1 次）
候选 20 篇文档
    ↓ Reranker 精确评分（20 对 query-doc 同时输入）
最终 3 篇 → 送入 LLM
```

**两阶段的平衡：** 召回阶段宁愿多取（不怕有噪声），精排阶段严格筛选。

---

## 代码实现

```python
from dashscope import TextReRank

class QianwenReranker:
    def rerank_documents(self, query, documents, top_k=3):
        contents = [doc.page_content for doc in documents]

        response = TextReRank.call(
            model="qwen3-rerank",
            query=query,
            documents=contents   # 批量发送所有候选文档
        )

        if response.status_code == 200:
            results = response.output["results"]
            # results 按相关性降序排列，每条有 index（原始位置）和 relevance_score
            ranked_indices = [result["index"] for result in results[:top_k]]
            return [documents[i] for i in ranked_indices]
```

**返回结构示例：**

```json
{
  "results": [
    {"index": 2, "relevance_score": 0.94},  ← 原来第3条，精排后升为第1
    {"index": 0, "relevance_score": 0.87},
    {"index": 4, "relevance_score": 0.71}
  ]
}
```

---

## 精排前后对比

假设 RRF 融合后的候选列表：

```
候选 1：关于"镜头选型的通用原则"（向量相近，但不精确）
候选 2：关于"焦距 f 值的计算公式"（命题精确，但向量稍远）← 用户真正想要的
候选 3：关于"相机分辨率与像素大小的关系"
```

向量检索排名：1 > 3 > 2（因为"选型原则"向量和问题最近）  
Reranker 精排：2 > 1 > 3（因为交叉编码器看到了"焦距计算公式"与问题的精确语义匹配）

---

## 主项目中的调用时机

```python
def search(self, query, k=5, llm=None):
    # ... FAISS 检索 + BM25 检索 + RRF 融合 ...

    # 精排：候选数 > 1 时才有意义
    if self.reranker and len(results) > 1:
        try:
            results = self.reranker.rerank_documents(query, results)
            print(f"Rerank 完成，保留前 {len(results)} 条")
        except Exception as e:
            print(f"Rerank 失败，使用原始结果: {e}")  # 精排失败不影响主流程

    return results
```

**降级策略：** Reranker API 出错时，直接使用 RRF 融合结果，系统继续正常运行。

---

## 成本与性能权衡

| 方案 | 精度 | 延迟 | API 成本 |
|------|------|------|---------|
| 只用 FAISS | 中 | ~50ms | Embedding 费用 |
| FAISS + BM25 + RRF | 较高 | ~80ms | Embedding 费用 |
| + Reranker | 高 | ~300ms | + Reranker 费用 |

对于问答系统，用户愿意等 300ms 换取更准确的答案，成本是合理的。  
如果是实时推荐场景（毫秒级要求），可以关闭 Reranker。

---

## 动手思考

1. 如果知识库只有 50 篇文档，还需要 Reranker 吗？
2. 精排只在 `len(results) > 1` 时启用，只有 1 条候选时直接返回——为什么？
3. `top_k` 设为 3 意味着最多 3 条文档进入 LLM 的 Prompt，这个数字如何影响回答质量？
