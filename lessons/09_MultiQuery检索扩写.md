# 09 · MultiQuery 检索扩写

> 对应主项目 `knowledge_base.py` 中的 `_build_multi_query_retriever()`

---

## 问题：一个问题问不全

用户的提问往往是口语化的，不一定能覆盖知识库中所有相关的表述角度：

```
用户问：镜头选型的计算方法

知识库中有：
  ✅ "镜头选型的计算方法"         ← 向量检索命中
  ✅ "如何根据视野和距离选焦距"   ← 可能命中
  ❌ "工业相机镜头参数计算公式"   ← 向量距离稍远，可能被阈值过滤
  ❌ "FOV WD 焦距三角关系"        ← 用了缩写，很可能漏掉
```

单次检索只用原始问题，召回率受限于问题的表述方式。

---

## 解决思路：让 LLM 改写问题

**MultiQueryRetriever** 的思路：用 LLM 把原始问题从多个角度改写成几个不同的查询语句，  
用所有查询语句分别检索，合并去重后得到更全面的候选集。

```
原始问题：镜头选型的计算方法
    ↓ LLM 改写
查询1：如何根据视野和工作距离选择焦距
查询2：工业相机镜头参数计算公式
查询3：FOV WD 焦距关系
    ↓ 4 个查询（含原始）分别检索 FAISS
    ↓ 合并去重
候选集（覆盖面更广）
    ↓ RRF + Reranker
最终结果
```

---

## 代码实现

### 改写提示词

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "你是一位信息检索专家。请为以下问题生成3个不同角度的查询语句，"
        "帮助从知识库中检索到更全面的信息。\n"
        "原始问题：{question}\n"
        "直接输出3个查询语句，每行一个，不要编号或其他格式："
    )
)
```

### 构建 MultiQueryRetriever

```python
from langchain.retrievers import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,   # 底层仍是 FAISS
    llm=llm,                    # 用来改写问题的 LLM
    prompt=prompt,
    include_original=True       # 保留原始问题的检索结果
)

# 使用方式和普通 retriever 完全一样
results = retriever.invoke("镜头选型的计算方法")
```

LangChain 内部会自动：
1. 调用 LLM 生成 3 个改写查询
2. 4 个查询（含原始）并行检索
3. 合并结果，自动去重

---

## 性能与成本权衡

| 方案 | 额外 LLM 调用 | 召回提升 | 额外延迟 |
|------|-------------|---------|---------|
| 单次检索 | 0 次 | 基准 | 0ms |
| MultiQuery（3 改写） | 1 次（轻量） | 显著提升 | +100~200ms |

**适合开启的场景：**
- 用户问题较短、口语化、容易漏关键词
- 知识库文档用词专业、与用户习惯表述差距大

**可以关闭的场景：**
- 对响应延迟要求极高（毫秒级）
- 知识库文档本身就是 Q&A 格式（问题表述高度一致）

---

## 主项目中的开关控制

```python
# config.py
MULTI_QUERY_RETRIEVAL = True   # 改为 False 即可关闭

# knowledge_base.py
if Config.MULTI_QUERY_RETRIEVAL and llm:
    retriever = self._build_multi_query_retriever(base_retriever, llm)
else:
    retriever = base_retriever
```

`llm` 参数从 `app.py` 中的 `kb.search(query, llm=llm.llm)` 传入，  
CLI 模式（`main.py`）调用的是 `kb.search(query)`，不传 `llm`，自动退化为单次检索。

---

## 注意事项

**问题：** LLM 改写的查询有时会偏离原意或产生幻觉，导致检索到不相关内容。

**缓解措施：**
- 提示词中强调"不同角度"而非"不同问题"
- 后续的 Reranker 精排会淘汰不相关内容
- `include_original=True` 确保原始问题的检索结果不会丢失

---

## 动手思考

1. 用你自己的问题测试：把问题交给 LLM 改写 3 个版本，对比哪个版本检索结果最好
2. 如果 LLM 改写时产生了与原问题完全无关的查询，最终结果会受影响吗？（提示：Reranker）
3. 为什么提示词说"直接输出，不要编号"？如果有编号会发生什么？
