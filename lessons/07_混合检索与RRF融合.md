# 07 · 混合检索与 RRF 融合

> 对应主项目 `knowledge_base.py` 中的 `search()` 和 `_rrf_merge()`

---

## 问题：两路结果如何合并？

FAISS 返回了 5 条结果，BM25 也返回了 5 条结果，两个列表有重叠也有各自独有的内容。  
合并时面临两个问题：

1. **评分不可比：** FAISS 用余弦相似度（0~1），BM25 用 TF-IDF 加权分（无上界），两者量纲不同，无法直接相加
2. **排名如何融合：** FAISS 认为第 1 名最好，BM25 也有自己的第 1 名，谁更重要？

---

## RRF：倒数排名融合

**Reciprocal Rank Fusion（RRF）** 的思路极其优雅：  
**不关心分数，只关心排名**——排名越靠前，得分越高，两路都靠前的文档得分叠加。

### 公式

$$\text{RRF}(d) = \sum_{\text{每路检索}} \frac{1}{k + \text{rank}(d)}$$

- `rank(d)`：文档 d 在该路检索中的排名（从 1 开始）
- `k`：常数，默认 60，防止排名 1 的权重过于极端

### 手动计算示例

假设 FAISS 和 BM25 各返回 5 条结果：

```
FAISS 排名：  文档A(1) 文档B(2) 文档C(3) 文档D(4) 文档E(5)
BM25 排名：   文档C(1) 文档A(2) 文档F(3) 文档B(4) 文档G(5)
```

计算每个文档的 RRF 分（k=60）：

| 文档 | FAISS 得分 | BM25 得分 | RRF 总分 |
|------|-----------|----------|---------|
| 文档A | 1/(60+1)=0.0164 | 1/(60+2)=0.0161 | **0.0325** |
| 文档B | 1/(60+2)=0.0161 | 1/(60+4)=0.0156 | **0.0317** |
| 文档C | 1/(60+3)=0.0159 | 1/(60+1)=0.0164 | **0.0323** |
| 文档D | 1/(60+4)=0.0156 | 不在列表 = 0 | 0.0156 |
| 文档F | 不在列表 = 0 | 1/(60+3)=0.0159 | 0.0159 |

**RRF 最终排名：** 文档A > 文档C > 文档B > 文档F > 文档D

**结论：** 两路都命中的文档（A、B、C）分数叠加，排名靠前；只有一路命中的文档（D、F）分数较低。

---

## 代码实现

```python
def _rrf_merge(self, faiss_docs, bm25_docs, k, rrf_k=60):
    scores = {}   # key: 文档前80字（去重用）  value: RRF 分数
    doc_map = {}  # key: 文档前80字            value: Document 对象

    for rank, doc in enumerate(faiss_docs):
        key = doc.page_content[:80]
        scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:80]
        scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        doc_map[key] = doc

    # 按 RRF 分数降序排列
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:k]]
```

---

## 为什么 k=60？

k 值越大，排名靠前的文档优势越小（分数差距越小）。  
k=60 是 RRF 原始论文建议的默认值，实践中表现稳健。

```
k=1：  rank1 得 0.5，rank2 得 0.33，差距极大（排名权重过强）
k=60： rank1 得 0.0164，rank2 得 0.0161，差距温和（多路意见均被考虑）
k=∞：  所有排名得分趋同（退化为简单计数）
```

---

## 完整检索流程

```
用户问题
    ↓
┌──────────────────────────────────────┐
│  并行检索（ThreadPoolExecutor）        │
│  FAISS 语义检索（带相似度阈值过滤）    │
│  BM25  关键词检索                     │
└──────────────────────────────────────┘
    ↓
RRF 融合（按排名打分，不依赖原始分数）
    ↓
去重（前80字相同视为同一文档）
    ↓
Reranker 精排（下一节）
    ↓
返回前 RERANK_TOP_K 条
```

---

## 动手思考

1. 为什么用"文档前80字"做去重键而不是完整内容？
2. 如果三路检索（FAISS + BM25 + 标题匹配），RRF 公式如何扩展？
3. k=0 时 rank1 的文档得分是多少？这会导致什么问题？

---

## 主项目日志解读

运行 `app.py` 后，每次检索会输出：

```
KB 命中：FAISS=4 BM25=6
Rerank 完成，保留前 3 条
```

- `FAISS=4`：通过相似度阈值过滤后，有 4 条语义相关结果
- `BM25=6`：关键词检索命中 6 条（TOP_K×2）
- 两路经 RRF 融合后去重，交给 Reranker 精排保留最终 3 条
