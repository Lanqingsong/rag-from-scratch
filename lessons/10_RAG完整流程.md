# 10 · RAG 完整流程

> 本节串联前 9 节所有概念，对应主项目全部文件

---

## 什么是 RAG？

**RAG（Retrieval-Augmented Generation，检索增强生成）** 解决的是大模型的核心局限：

| 问题 | 说明 |
|------|------|
| 知识截止 | 模型训练后的新事件、新文档它不知道 |
| 私有数据 | 你公司的内部文档它没见过 |
| 幻觉 | 模型可能编造听起来合理但错误的答案 |

**RAG 的思路很直接：**

```
不让模型"记住"所有知识，而是在回答时"查阅"相关文档
→ 模型只负责理解和生成语言，事实来源于你控制的文档库
```

---

## 完整流程总览

### 阶段一：离线构建（运行一次）

```
原始文档（PDF / MD / TXT）
    ↓ 1. 文档加载（DirectoryLoader）
原始文本
    ↓ 2. 文本切分（MarkdownSplitter / RecursiveSplitter / ...）
chunks（500~1200 字的文本片段）
    ↓ 3. 向量化（QianwenEmbeddings）
向量列表
    ↓ 4. 建立索引（FAISS + BM25）
本地存储（vector_store/）
```

### 阶段二：在线检索（每次问答）

```
用户问题
    ↓ 5. 问题向量化（同一 Embedding 模型）
问题向量
    ↓ 6. 双路检索（FAISS 语义 + BM25 关键词）
候选文档（10~15 条）
    ↓ 7. RRF 融合排名
    ↓ 8. Reranker 精排
精选文档（3 条）
    ↓ 9. 注入 Prompt
    ↓ 10. LLM 生成（流式输出）
最终回答
```

---

## 代码文件与流程的对应关系

```
config.py          → 参数中枢：所有阈值、模型名、路径都在这里
    ↓
knowledge_base.py  → 流程 1~8：文档加载、切分、向量化、双路检索、RRF、Reranker
splitters.py       → 流程 2 的具体实现：5 种切分策略
llm_client.py      → 流程 9~10：Prompt 模板选择 + LLM 调用
prompts.py         → 流程 9 的 Prompt 模板：KB_ONLY / DUAL_SOURCE / WEB_ONLY / GENERAL
web_search.py      → 可选的第 11 步：SearXNG 联网补充（熔断器保护）
app.py             → 总调度：接收 HTTP 请求，协调以上所有模块，SSE 流式推送结果
```

---

## 关键设计决策解析

### 决策 1：相似度阈值（KB_RELEVANCE_SCORE = 0.62）

```python
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.62, "k": 5}
)
```

**为什么需要阈值？**  
FAISS 总会返回最相近的 k 条，即使相似度只有 0.3（完全不相关）。  
没有阈值时，"今天天气怎么样"也会检索到机器视觉文档，污染 LLM 的上下文。

**调参建议：**
- 阈值太高（> 0.8）：大量相关文档被过滤，系统倾向于回答"知识库中没有相关信息"
- 阈值太低（< 0.5）：噪声文档进入上下文，LLM 被干扰，回答质量下降
- 推荐从 0.6 开始，根据实际效果微调

---

### 决策 2：4 种 Prompt 模式

```python
# llm_client.py 根据检索结果自动选择
def _pick(self, kb, web):
    if kb and web:   return self.dual_chain    # 本地 + 网络都有
    if kb:           return self.kb_chain      # 只有本地知识库
    if web:          return self.web_chain     # 只有网络搜索
    return self.general_chain                  # 两者都没有，直接问 LLM
```

**为什么要分 4 种？**  
注入不存在的 `{context}` 占位符会报错；没有上下文时不应该在 Prompt 里提及"参考资料"。  
每种模式使用精确匹配的 Prompt 模板，避免模板变量错配。

---

### 决策 3：对话引用检测

```python
# app.py
if _is_dialogue_ref(query, history):
    # 直接走 LLM（用历史），跳过检索
    for chunk in llm.stream(query, None, None, history):
        yield _sse({'type': 'token', 'content': chunk})
```

用户说"上一个问题是什么"或"你刚才说的那个公式"——这类问题的答案在对话历史里，  
强行检索知识库不仅浪费时间，还可能引入不相关内容干扰 LLM 理解。

---

### 决策 4：SSE 流式输出的分阶段推送

```
① status 事件  → "正在检索知识库和互联网..."（用户看到进度）
② refs 事件   → 参考资料列表（检索完成，LLM 还没开始，先让用户看来源）
③ token 事件  → 逐 token 推送（用户看到实时生成过程）
④ done 事件   → 完整的 Markdown 渲染 HTML（前端替换原始文本）
```

这个顺序的设计让用户感知到"系统在积极工作"，减少等待焦虑。

---

## 端到端追踪一次问答

以"如何计算镜头焦距"为例，追踪系统内部的完整执行过程：

```
1. app.py 收到 POST /api/chat/stream
   query = "如何计算镜头焦距"
   history = [...]

2. _is_dialogue_ref() 返回 False → 进入正常检索流程

3. _parallel_search() 并行启动：
   线程1: kb.search("如何计算镜头焦距", llm=llm.llm)
     → MultiQuery 改写: ["焦距f值计算公式", "WD sensor FOV关系", "工业镜头选型参数"]
     → FAISS 检索 4 个查询，各返回 5 条，合并去重 → 12 条候选
     → BM25 检索原始问题 → 6 条候选
     → RRF 融合 → 排序后取前 10 条
     → Reranker 精排 → 保留前 3 条
   线程2: web_searcher.search("如何计算镜头焦距")
     → SearXNG 返回 3 条网页摘要

4. 推送 refs 事件（3 条本地 + 3 条网络）

5. llm._pick(kb=3条, web=3条) → dual_chain（本地+网络双源模式）

6. 构建 Prompt：
   system: "你是机器视觉工程师AI助手...
            ## 本地资料
            2.1 焦距计算 f = WD × sensor_size / FOV ...（3条）
            ## 最新资讯
            [网页标题] 工业镜头选型指南... （3条）"
   human: "如何计算镜头焦距"

7. DeepSeek 流式生成，逐 token 推送 token 事件

8. 生成完毕，推送 done 事件（含 Markdown 渲染 HTML）
```

---

## 常见问题与调优

| 症状 | 可能原因 | 调整方向 |
|------|---------|---------|
| 总说"知识库中没有相关信息" | 阈值太高或文档未正确入库 | 降低 `KB_RELEVANCE_SCORE`，检查 vector_store |
| 回答照搬原文、缺乏整合 | system prompt 指令不够强 | 修改 `prompt_templates/system.txt` |
| 检索到不相关文档 | 阈值太低，或切分粒度太粗 | 提高阈值，优化切分策略 |
| 专有名词经常漏检 | BM25 未安装或禁用 | `pip install rank_bm25` |
| 回答正确但速度慢 | MultiQuery + Reranker 叠加延迟 | 关闭 `MULTI_QUERY_RETRIEVAL` 或降低 `TOP_K` |

---

## 恭喜完成学习路径！

你现在理解了一个生产级 RAG 系统的所有核心组件。  
建议接下来：

1. **阅读 `knowledge_base.py`** 完整代码，逐行对应本教程的概念
2. **阅读 `app.py`**，理解 Flask 路由和 SSE 流式输出的实现
3. **替换知识库内容**，用你自己领域的文档测试系统效果
4. **调整参数**（阈值、chunk_size、TOP_K），观察对检索质量的影响
5. **修改提示词**（`prompt_templates/system.txt`），适配你的业务场景
