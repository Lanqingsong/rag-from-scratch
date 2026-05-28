# 从零搭建基于 LangChain 的知识库问答系统：工程实践记录

> **GitHub**：[https://github.com/Lanqingsong/rag-from-scratch](https://github.com/Lanqingsong/rag-from-scratch)  
> **技术栈**：LangChain · DeepSeek · FAISS · DashScope · Flask · rank-bm25 · SearXNG

---

# 1. 动机：为什么不用现成工具，要自己写

## 1.1 场景背景

项目的原始需求很具体：机器视觉领域的内部知识库问答。文档类型包括相机选型手册、镜头计算指南、工艺参数表、设备标定流程等，用户是一线工程师，问题很精确——"CS-040GX 的最大帧率是多少"、"GigE 接口的带宽上限"这类问题比"什么是机器视觉"多得多。

早期验证阶段，用千问 Embedding 把文档向量化存进 FAISS，检索 top-5 喂给 DeepSeek，Demo 跑起来没什么毛病。等真正到了工程师手里，问题就来了，这个之后再说。

在想"用什么搭"之前，先考虑的是"能不能直接用 Coze / Dify / FastGPT 这些工具"。

## 1.2 私有数据不适合上云

几百页技术手册里有设备选型数据、供应商价格参数、内部工艺标准，这些内容不适合上传到任何第三方 SaaS 平台。

这不是"我有没有信任他们"的问题，是合规层面的硬约束。Coze、Dify 的托管版本不满足要求，Dify 有私有化部署版，FastGPT 也有，但后面会说为什么最终还是选择了自己写。

## 1.3 用了这些工具，卡在了这几个地方

### 1.3.1 文档切分无法控制

这是最大的问题。

技术手册里有这样的结构：一个标题下面跟着若干参数行，比如"GigE 接口规格"下面列了带宽、延迟、帧率、分辨率四行数据。如果按固定字符数切分，这个标题和它下面的参数很可能被切成两个 chunk，检索"GigE 带宽上限"时，可能召回的是一个没有参数数值的 chunk，LLM 的回答自然就不对。

Dify 和 FastGPT 在文档处理阶段基本只有"按字符数切割"和"自动切割"两个选项，有些支持调整 chunk_size，但整体逻辑是固定的。遇到结构化程度高的文档，切出来的 chunk 质量很差，召回就没法保证。

自己写的话，可以针对 Markdown 文档按标题切，针对编号格式文档（1.1 xxx）用正则定边界，甚至对完全没有结构的文档用语义相似度来找自然断点。这个灵活度在集成工具里基本拿不到。

### 1.3.2 检索策略是黑盒

这些工具内部用的什么检索方案，调了哪些参数，召回不准时往哪个方向调，基本上不透明。

实际工程中，"向量检索 + BM25 关键词检索 + RRF 融合 + Reranker 精排"这个四层管道，每一层都有可调的参数，也都有它存在的具体原因。如果用黑盒工具，召回质量出问题时排查起来很困难，调优更是无从下手。

### 1.3.3 网络搜索和本地知识库的联动

工程师有时候问的是市面上新型号的参数，本地知识库里没有。这时候系统应该能自动搜索，并且在回答里同时参考本地知识和网络结果。

Coze 可以接搜索插件，但搜索和本地知识库之间的调度逻辑是固定的，不能根据业务需求自定义"什么时候用本地、什么时候补搜索、结果怎么合并"。自己写的话，这些逻辑完全在手里。

### 1.3.4 平台绑定和长期维护风险

Coze 是字节的产品，Dify 依赖其服务端。私有化部署版本的升级维护也是问题。自己写的代码，依赖链完全清楚，出了问题知道去哪查，也不存在某天平台政策变了要迁移的风险。

## 1.4 技术栈本身值得学透

还有一个原因不太"工程化"：LangChain 的 LCEL 管道、FAISS 的向量存储、BM25 的关键词检索、Reranker 的原理、SSE 流式推送——这些东西是大模型应用开发的核心构件，如果一直用包装好的工具，就只会点几下配置界面。

用来做这个项目的同时，把每一个技术决策的原因都记录在 `lessons/` 目录里，写了 19 篇教学文档。代码和文档同步更新，既是项目记录，也是学习过程。

---

# 2. 整体架构

## 2.1 组件选型

| 组件 | 选型 | 选它的原因 |
|------|------|-----------|
| LLM | DeepSeek | 中文理解好，API 价格合理，兼容 OpenAI SDK |
| Embedding | 阿里云千问 text-embedding-v2 | 中文语料训练，支持批量调用 |
| Reranker | 千问 qwen3-rerank | 和 Embedding 同一生态，调用方式统一 |
| 向量库 | FAISS | 纯文件，无需启动服务，速度快 |
| 关键词检索 | rank-bm25 | 轻量，纯 Python，不需要外部服务 |
| LLM 框架 | LangChain | LCEL 管道语法，MultiQueryRetriever |
| Web 框架 | Flask | SSE 流式输出简洁，现有代码同步友好 |
| 网络搜索 | SearXNG | 自托管，不依赖第三方 API 限额 |

## 2.2 一次查询的完整流程

用户发来一个问题，系统按以下顺序处理：

1. **对话引用检测**：判断这个问题是不是指向当前对话（"你刚才说的那个公式"、"继续"），如果是，跳过检索，直接带历史发给 LLM。
2. **并行检索**：KB 检索和网络搜索同时发起，互不等待。
   - KB 检索内部：FAISS 语义检索 → BM25 关键词检索 → RRF 融合 → Reranker 精排
3. **先推参考资料**：检索完成后立刻通过 SSE 把参考来源推给前端，不等 LLM 开始回答。
4. **流式输出**：LLM 开始生成，逐 token 通过 SSE 推给前端，首 40 个字符缓冲以过滤套话开头。
5. **Markdown 渲染**：LLM 回答完毕，服务端渲染为 HTML，通过最后一个 SSE 事件推给前端。

---

# 3. 文档切分：这是整个系统质量的起点

## 3.1 chunk_size 设多大不是核心问题

大多数 RAG 教程花大量篇幅讨论 chunk_size 应该设 512 还是 1024，这个参数当然有影响，但对于结构化程度高的技术文档，**在哪里切**比**切多大**重要得多。

举个例子：一份机器视觉相机的技术规格文档，结构是这样的：

```
## GigE Vision 接口规格

- 传输带宽：1 Gbps（GigE）/ 10 Gbps（10GigE）
- 触发延迟：< 1 μs
- 支持协议：GigE Vision 2.0
- 典型工作距离：0.5 m ~ 5 m
```

如果用固定字符数切割，"GigE Vision 接口规格"这个标题可能和上一节末尾的内容拼在一起，而这节的参数数据被切走了。用户问"GigE 触发延迟是多少"，召回的 chunk 里没有参数值，LLM 只能说"没有相关信息"。

问题不是 chunk_size，是切割没有尊重文档的语义边界。

## 3.2 五种切分策略

项目里实现了五种策略，放在 `splitters.py` 里，通过 `kb_config.json` 配置。

### 3.2.1 Markdown 标题切割（MarkdownHeadingSplitter）

按 `##` / `###` / `####` 标题行切割，每个标题和它下面的内容作为一个 chunk。适合有良好 Markdown 结构的技术手册、教程文档。

超长 chunk（超过 chunk_size）会尝试按下一级标题二次切割，二次切割还是太长的话回退到递归字符切割。

```json
// kb_config.json
{
  "strategy": "markdown",
  "heading_level": 3,
  "chunk_size": 1000
}
```

`heading_level: 3` 表示按 `###` 切，`2` 按 `##` 切，依此类推。

### 3.2.2 自定义正则边界（RegexBoundarySplitter）

有些文档格式是编号式的，比如 `1.1 焦距计算`、`Q12: 如何标定相机`。这类文档有明确的逻辑边界，但不是 Markdown 标题，用正则匹配边界更准确。

```json
{
  "strategy": "regex",
  "pattern": "\\d+\\.\\d+\\s+[^\\n]+"
}
```

这个配置会识别 `1.1 xxx`、`2.3 xxx` 格式的行作为 chunk 起点，每个编号下的内容作为一个 chunk。

### 3.2.3 语义切割（SemanticSplitter）

基于相邻句子的 Embedding 余弦相似度找自然断点：相似度骤降的地方说明话题在这里切换，就在这里切。用的是 LangChain 的 `SemanticChunker`。

这个策略最贵，因为每对相邻句子都要计算一次 Embedding，适合完全没有结构的自由文本（比如产品说明、工艺描述段落）。代码里做了异常保底，语义切割失败时自动回退到递归切割。

```json
{
  "strategy": "semantic",
  "threshold_type": "percentile",
  "threshold_value": 95
}
```

`threshold_value: 95` 表示在相似度下降最大的 5% 处切割，值越大切出来的块越多。

### 3.2.4 递归字符切割（RecursiveSplitter）

LangChain 的 `RecursiveCharacterTextSplitter`，按分隔符层级递归切割：`\n\n` → `\n` → `。` → `！` → `？` → `；` → 空格。遇到任何前面策略不适用的文档，这个是最终保底。

### 3.2.5 Auto 自动探测（AutoSplitter）

不想为每个目录手写配置的话，`auto` 策略会检测每份文档的结构并自动选策略：

- 文档里有 3 个及以上 `###` 标题 → 用 `markdown(heading_level=3)`
- 有 3 个及以上 `##` 标题 → 用 `markdown(heading_level=2)`
- 有 3 个及以上 `1.1 xxx` 格式行 → 用 `regex`
- 都没有 → 用 `recursive`

检测逻辑：

```python
def _detect_structure(text):
    if len(re.findall(r'^###\s+.+', text, re.M)) >= 3:
        return 'markdown', {'heading_level': 3}
    if len(re.findall(r'^##\s+.+', text, re.M)) >= 3:
        return 'markdown', {'heading_level': 2}
    if len(re.findall(r'^\d+\.\d+\s+\S+', text, re.M)) >= 3:
        return 'regex', {'pattern': r'\d+\.\d+\s+[^\n]+'}
    return 'recursive', {}
```

## 3.3 per-directory 配置：不同目录用不同策略

知识库目录结构可以是这样的：

```
knowledge_base/
├── cameras/
│   ├── kb_config.json      ← strategy: markdown
│   └── basler_ace2.md
├── lenses/
│   ├── kb_config.json      ← strategy: regex
│   └── lens_selection.md
└── general/                ← 无配置文件，使用 auto
    └── faq.txt
```

每个子目录独立配置，系统启动时遍历每个目录，读取各自的 `kb_config.json`，用对应的策略切分后统一建索引。

---

# 4. 混合检索：四层过滤管道

## 4.1 纯向量检索的两个硬伤

Embedding 模型的训练目标是语义相似度，"相机分辨率"和"图像像素数量"在向量空间里距离近，因为它们在大量文本里经常共现。

但这带来两个问题：

**问题一：专有名词召回失败。**  
"CS-040GX"这种型号编码在训练语料里极少出现，模型对它几乎没有语义记忆，向量表示基本是随机方向。用户输入这个型号，FAISS 检索余弦相似度低于阈值，直接返回 0 条结果。

**问题二：相关度阈值难以设准。**  
阈值设高了，正确答案被过滤掉；设低了，不相关的文档也进来，LLM 的回答质量下降。这个阈值在不同文档、不同问法下表现不一致，调起来很头疼。

## 4.2 BM25 补刀关键词盲区

### 4.2.1 BM25 和 TF-IDF 的区别

BM25 是 TF-IDF 的改进版，核心加了两个机制：词频饱和（一个词出现 10 次不代表相关度是出现 1 次的 10 倍）和文档长度归一化（长文档不占便宜）。

$$\text{score}(D, Q) = \sum_i \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}$$

其中 $k_1$ 控制词频饱和（默认 1.5），$b$ 控制长度归一化（默认 0.75）。对"CS-040GX"这个查询，BM25 直接做字符串匹配，精确找到包含这个型号的文档，不依赖任何语义理解。

### 4.2.2 RRF 把两路结果合并

FAISS 和 BM25 各自返回一个排序列表，用 Reciprocal Rank Fusion（RRF）合并：

```python
def _rrf_merge(self, faiss_docs, bm25_docs, k, rrf_k=60):
    scores = {}
    doc_map = {}
    for rank, doc in enumerate(faiss_docs):
        key = doc.page_content[:80]
        scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        doc_map[key] = doc
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:80]
        scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        doc_map[key] = doc
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:k]]
```

公式 `1/(k + rank)` 中，`k=60` 来自 Cormack 等人 2009 年的论文，实验发现这个值在多种任务上稳健——它给排名靠前的文档适度奖励，同时不会把排名靠后的文档完全压死。两路都检索到的文档得分叠加，只有一路命中的也能参与排名。

## 4.3 Reranker：召回完了再精排

### 4.3.1 为什么要有这一层

即使召回准了，还有个问题：把 10 条参考文档塞给 LLM，LLM 会重点看开头和结尾，中间的内容容易被忽略。斯坦福 2023 年的 [Lost in the Middle](https://arxiv.org/abs/2307.03172) 论文做了实验：正确答案在第 1 条或最后一条时，准确率约 70%；放在第 6\~9 条时，降至约 40%。

所以召回的目标不仅是"包含正确答案"，还要"让正确答案排在前面"。

### 4.3.2 双编码器 vs 交叉编码器

Embedding 检索用的是**双编码器（Bi-encoder）**：查询和文档各自独立编码成向量，然后计算余弦相似度。好处是速度快，一亿个文档的向量可以离线预计算好；坏处是编码时两者互不可见，捕捉不到细粒度的语义交互。

Reranker 用的是**交叉编码器（Cross-encoder）**：把查询和候选文档拼接在一起送进同一个模型，每一个 attention 层都能看到两者的交互，直接输出一个精确的相关性分数。代价是慢，因为每个候选文档都要重新过一遍，不能预计算。

两者配合使用——FAISS 快速拿到 10\~20 个候选，Reranker 精排保留 top-3：

```python
def rerank_documents(self, query, documents, top_k=Config.RERANK_TOP_K):
    contents = [doc.page_content for doc in documents]
    from dashscope import TextReRank
    response = TextReRank.call(
        model='qwen3-rerank',
        query=query,
        documents=contents
    )
    if response.status_code == 200:
        results = response.output['results']
        ranked_indices = [result['index'] for result in results[:top_k]]
        return [documents[i] for i in ranked_indices]
```

### 4.3.3 什么时候调用 Reranker

Reranker 有额外的 API 费用，当召回文档数 ≤ 1 时调用没有意义。代码里的判断是：`if self.reranker and len(results) > 1`，超过 1 条才精排。

## 4.4 MultiQueryRetriever：查询改写扩大覆盖面

用户的表述方式多种多样，"GigE 接口最大带宽"和"千兆以太网相机的传输速率上限"在语义上等价，但向量空间里距离未必很近。

`MultiQueryRetriever` 让 LLM 把用户的原始问题改写成 3 个不同角度的查询，三路并行检索，结果取并集。对于专业词汇多、表述变体丰富的场景，命中率提升明显。

这里踩过一个坑：项目用的 LangChain 1.3.1，这个版本的 `MultiQueryRetriever` 在 `langchain_classic` 包里，而不是 `langchain.retrievers`。导入链需要写成：

```python
try:
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    try:
        from langchain.retrievers.multi_query import MultiQueryRetriever
    except ImportError:
        from langchain.retrievers import MultiQueryRetriever
```

按优先级逐级尝试，保证在不同版本环境下都能找到。

---

# 5. 多轮对话的两个难点

## 5.1 对话引用识别：这个问题不用去检索知识库

### 5.1.1 问题是怎么出现的

用户问完"镜头焦距怎么计算"，接着问"那工作距离呢"。系统拿着"那工作距离呢"去检索知识库，当然什么都找不到，然后回答"没有找到相关资料"。

这类问题的语义锚点在对话历史里，不在知识库里。拿去检索是错误的操作。

### 5.1.2 识别方案

在进入检索流程前，先判断这个问题是不是对话引用：

```python
_DIALOGUE_REF = re.compile(
    r'(上一个|上一条|上一轮|上面|刚才|之前(说|问|讲|提到?)?|你说|你提到|'
    r'你刚|你之前|继续|你上面|你前面|解释一下刚|你说的)',
    re.IGNORECASE
)

def _is_dialogue_ref(query: str, history: list) -> bool:
    if not history:          # 无历史时不可能是引用
        return False
    if _DIALOGUE_REF.search(query):
        return True
    if len(query) <= 15 and re.search(r'(这个|那个|它|这些|那些)', query):
        return True          # 短问题 + 指代词 → 大概率引用
    return False
```

命中时，跳过检索，直接把完整对话历史发给 LLM。

### 5.1.3 局限性

正则匹配有误检和漏检。"继续"这个词可能是让系统继续解释，也可能是一个全新的问题（"继续教育是什么"）。短问题加指代词的规则也会有偏差。

更健壮的方案是用小模型做意图分类（query + last_turn → {dialogue_ref | fresh_query}），但这意味着每次请求多一次 LLM 调用，对当前体量来说成本增加不小。目前接受这个精度，日志里记录每次判断结果，后续根据错误频率再决定是否升级。

## 5.2 Token 预算：历史记录怎么截断

对话历史不能无限累积，否则超出上下文窗口。这里的截断策略是**从最新一轮往前截，超出预算就停**，而不是从最早一轮往后截：

```python
items = list(reversed(history_data or []))
total_tokens = 0
selected = []
for item in items:
    t = _count_tokens(item.get('content', ''))
    if total_tokens + t > Config.MAX_HISTORY_TOKENS:
        break
    selected.insert(0, (item['role'], item['content']))
    total_tokens += t
```

逻辑很直接：最近几轮对话对理解当前问题最有价值，历史太长时优先丢弃早期的轮次。MAX_HISTORY_TOKENS 默认 3000。

Token 计数用 tiktoken 的 `cl100k_base` 编码，tiktoken 不可用时降级到字符数的一半（中文约 2 字/token）。

---

# 6. 网络搜索：知识库之外的兜底

## 6.1 并行检索，不串行等待

网络搜索和知识库检索是并行发起的，用 `ThreadPoolExecutor`：

```python
def _parallel_search(query):
    if web_searcher:
        with ThreadPoolExecutor(max_workers=2) as ex:
            kb_f  = ex.submit(kb.search, query, Config.TOP_K_RESULTS, llm.llm)
            web_f = ex.submit(web_searcher.search, query)
            return kb_f.result(timeout=20), web_f.result(timeout=15)
    return kb.search(query, llm=llm.llm), []
```

知识库检索超时 20 秒，网络搜索超时 15 秒，互不阻塞。网络搜索用的是 SearXNG 自托管实例，不依赖任何第三方 API 限额。

## 6.2 熔断器防止搜索服务雪崩

SearXNG 不稳定时，如果每次请求都等超时，整个响应链路就会被拖慢。熔断器在连续失败 3 次后自动"断路"，后续请求直接跳过网络搜索，60 秒后自动尝试恢复：

```python
class _CircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_timeout=60):
        self._failures  = 0
        self._threshold = failure_threshold
        self._timeout   = recovery_timeout
        self._opened_at = None
        self._lock      = Lock()

    @property
    def is_open(self):
        with self._lock:
            if self._opened_at is None:
                return False
            if time.time() - self._opened_at > self._timeout:
                self._opened_at = None
                self._failures  = 0
                return False
            return True
```

搜索失败时还有指数退避重试（1s、2s、4s），三次都失败才触发熔断器计数。这是 Martin Fowler 的 Circuit Breaker 模式，原文见 [martinfowler.com/bliki/CircuitBreaker.html](https://martinfowler.com/bliki/CircuitBreaker.html)。

---

# 7. 流式输出的工程问题

## 7.1 为什么选 SSE 不选 WebSocket

SSE（Server-Sent Events）是单向的：服务端推，客户端收。WebSocket 是双向的，但这个场景不需要双向。SSE 更轻，在 Flask 里实现极简：

```python
return Response(generate(), mimetype='text/event-stream',
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
```

FastAPI 的 SSE 需要额外装 `sse-starlette`，还要处理异步生成器的兼容问题。Flask 这个写法就够用了。

## 7.2 开头套话的拦截方案

LLM 经过 RLHF 训练后形成了一个习惯：在有参考资料时，先说"根据知识库内容，..."。这是模型认为"负责任的引用"，但用户看到这句话感觉是在被敷衍。

改 Prompt 有效果，但不持久——不同温度、不同版本的模型，这个行为强度不一样，改一段时间效果又退化。

流式输出是逐 token 推送的，第一个 token 就是"根"。要拦截开头，需要在服务端设一个缓冲区，等攒够了再决定要不要清洗：

```python
BUF_SIZE = 40  # 足够覆盖最长的"根据..."开头

buf = ""
buf_flushed = False

for chunk in llm.stream(...):
    if not chunk:
        continue
    if not buf_flushed:
        buf += chunk
        if len(buf) >= BUF_SIZE:
            buf = _BAD_OPENER.sub('', buf, count=1).lstrip()
            buf_flushed = True
            yield _sse({'type': 'token', 'content': buf})
    else:
        yield _sse({'type': 'token', 'content': chunk})

# 回答结束，剩余缓冲一次性推出（极短回答的 edge case）
if buf:
    buf = _clean(buf)
    yield _sse({'type': 'token', 'content': buf})
```

40 个字符是经验值，中文"根据您提供的知识库内容，"大约 15 个字，留 40 个字符缓冲足够判断开头是否需要清洗，又不会让用户感觉到明显的延迟。

`_BAD_OPENER` 正则：

```python
_BAD_OPENER = re.compile(
    r'^[\s\n]*(根据(提供的|以上|本|这些)?(知识库|资料|搜索结果|参考)(内容|信息|数据)?'
    r'[，,。]?\s*|知识库(中)?(提到|说明|显示|记录)[，,]?\s*)',
    re.IGNORECASE
)
```

## 7.3 参考资料和流式内容的推送时序

SSE 用不同的 `type` 字段区分事件类型：

- `status`：检索中提示
- `refs`：参考资料列表（检索完成后立刻推，不等 LLM）
- `token`：LLM 输出的每个 chunk
- `done`：LLM 输出完毕，携带服务端渲染好的 Markdown HTML
- `error`：异常

`refs` 在检索完成后立刻发出，用户可以在 LLM 还在生成时就看到参考来源，体验好一些。

---

# 8. Prompt 版本管理

## 8.1 没有版本控制是什么体感

Prompt 工程实际上是反复迭代的过程，改动频率比代码高得多。某次为了处理好"参数比较类"问题改了 `system.txt`，几天后发现"操作步骤类"问题的回答变差了，想回到之前的版本——只有一个被反复覆盖的文件，没有任何记录。

## 8.2 三层方案

**版本存档和回滚**：每次修改前先存档，存档文件名带时间戳。

```bash
# 存档当前版本
POST /api/prompts/save_version
# 返回 {"version": "system_20260528_143022.txt"}

# 激活历史版本（立即生效，无需重启）
POST /api/prompts/activate/system_20260528_143022.txt
```

**变量插值**：`system.txt` 里写占位符 `{{domain}}`、`{{user_type}}`，`variables.json` 里存实际值。这样调整定制内容不需要改主文件，也减少版本分叉。

```
# system.txt
你是 {{domain}} 领域的技术助手，服务对象是 {{user_type}}。
```

```json
// variables.json
{
  "system_vars": {
    "domain": "机器视觉",
    "user_type": "工程师"
  }
}
```

**热重载**：`POST /api/prompts/reload` 重建 LLM 链，修改提示词不需要重启 Flask 进程，不影响正在进行的对话。

---

# 9. 几个选型决定的思路

## 9.1 FAISS vs ChromaDB

ChromaDB 开箱即用体验好，有 Web UI，持久化也方便。选 FAISS 的原因：

- **速度**：FAISS 是 C++ 实现，Python 绑定极薄，几千个向量的检索延迟在毫秒级。ChromaDB 有额外的 HTTP 层开销。
- **无状态部署**：向量库是本地两个文件（`.faiss` + `.pkl`），整个知识库可以打包带走，部署到任何地方，不需要启动任何服务。

代价是没有 Web UI、不支持在线增量更新（需要重建），没有元数据过滤。当前场景不需要这些功能，所以 FAISS 的简单性反而是优势。

## 9.2 Flask vs FastAPI

FastAPI 的异步性能和自动 API 文档确实更好。选 Flask 主要是因为：

- SSE 流式输出在 Flask 里只需要 `Response(generator, mimetype='text/event-stream')`，非常简洁。FastAPI 的 SSE 要装 `sse-starlette`，还要把同步代码改成 `async`，或者用 `run_in_executor` 包装。
- 项目里大量模块是同步的（FAISS 检索、BM25 检索、文件操作），强行用异步框架需要额外处理，得不偿失。

如果以后要支持高并发（多用户同时检索），迁移到 FastAPI + 异步是合理的路径，LCEL 本身也支持 `ainvoke`。

## 9.3 LangChain 的争议与实际价值

LangChain 的口碑比较分裂，主要批评是：过度抽象、接口不稳定、版本间 breaking change 很多（这个确实踩过，`langchain_classic` 的 import 问题就是一例）。

但在这个项目里，它提供了三个真实价值：

- **LCEL 管道语法**：`Prompt | LLM | Parser` 这种写法让切换 LLM 提供商只需要换一个对象，不动其他逻辑。
- **MultiQueryRetriever**：查询改写功能，如果自己实现需要维护一段固定的 Prompt + 解析逻辑，LangChain 里直接用。
- **SemanticChunker**：语义切割的实现，算法本身不复杂，但有现成的就不用重复造。

前提是：用 LangChain 来用具体功能，不把整个应用逻辑都架在它的高层抽象上。核心的检索流程（FAISS + BM25 + RRF + Reranker）是自己控制的，LangChain 只负责几个有价值的组件。

---

# 10. 总结

这个项目从最早的单路向量检索，到现在的四层混合检索管道，每一步迭代都对应一个具体的工程问题：型号检索失败 → 加 BM25；LLM 忽视中间文档 → 加 Reranker；对话引用识别错误 → 加意图判断；开头套话 → 加流式缓冲过滤；Prompt 改了改不回来 → 加版本管理。

没有一个功能是"因为别人都这么做"加进来的，都有对应的问题驱动。

代码和 19 篇教学文档都在 GitHub 上，每一个设计决策在对应的文档里有更详细的推导过程：

> **[https://github.com/Lanqingsong/rag-from-scratch](https://github.com/Lanqingsong/rag-from-scratch)**

有问题或者更好的方案，欢迎提 Issue。

---

*本项目由 [lanqingsong874953727@outlook.com](mailto:lanqingsong874953727@outlook.com) 与 AI 助手协作开发。*

*编写时间：2026 年 5 月 28 日*
