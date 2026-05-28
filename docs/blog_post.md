# 我开源了一个生产级 RAG 知识库系统：四层检索管道 + 19 篇教学文档，附完整源码

---

## 项目信息

| | |
|---|---|
| **GitHub** | [https://github.com/Lanqingsong/rag-from-scratch](https://github.com/Lanqingsong/rag-from-scratch) |
| **开源协议** | MIT |
| **技术栈** | [LangChain](https://python.langchain.com/) · [DeepSeek](https://platform.deepseek.com/) · [FAISS](https://github.com/facebookresearch/faiss) · [rank-bm25](https://github.com/dorianbrown/rank_bm25) · [阿里云 DashScope](https://dashscope.console.aliyun.com/) · [Flask](https://flask.palletsprojects.com/) · [SearXNG](https://docs.searxng.org/) |

---

## 为什么要写这个项目

网上讲 RAG 的教程已经很多了，但我发现一个普遍现象：

**大多数教程在跑通 Demo 之后就停下来了。**

把文档切几刀、存进向量库、检索 top-K、塞进 Prompt，一个 `demo.py` 跑下来效果不错，作者得意地截个图，教程就结束了。

直到我真正把 RAG 系统交给用户用的时候，问题一个接一个出现：

**召回层**：文档里明明有 "qwen3-rerank" 这个型号，向量检索就是找不到。查了半天才发现——[Embedding 模型](https://platform.openai.com/docs/guides/embeddings)对精确名词天生不敏感，它更擅长理解语义，而不是精确匹配。

**精排层**：我把检索到的 10 条文档全塞给 LLM，以为信息越多越好，结果 LLM 反而忽略了中间最关键的那条。这个现象有个专业名字：**Lost in the Middle**（[论文链接](https://arxiv.org/abs/2307.03172)）。

**对话层**：用户问"你上面说的那个公式怎么推导"，系统乖乖跑去检索知识库，然后一脸茫然地说找不到相关内容。问题出在：这是对话引用，根本不应该检索。

**输出层**：每次流式回答都以"根据您提供的知识库内容……"开头。用户没用几次就烦了，我却不知道从哪里拦截这段话。

**提示词层**：Prompt 改了十几版，没有版本记录，某次改坏了，却不知道改之前是什么样的，只能靠记忆一点点还原。

**每一个问题都有具体的工程解法，但这些解法散落在论文、源码和零碎博客里，从来没有人把它们系统地放在一起。**

于是我花了几个月把这些解法一一实现，整理成这个项目，并写了 19 篇配套教学文档全部开源。

> 📌 **GitHub：[https://github.com/Lanqingsong/rag-from-scratch](https://github.com/Lanqingsong/rag-from-scratch)**  
> 如果对你有帮助，欢迎点个 ⭐ Star

---

## 系统架构

整个系统的处理流程如下：

```
用户提问
    ↓
[对话引用检测]
    ├── "你刚才说的..." → 直接从对话历史回答，跳过检索
    └── 普通问题 ↓
        ┌─────────────────────────────────────────┐
        │              并行检索层                  │
        │  FAISS 语义检索 + MultiQueryRetriever   │
        │  BM25 关键词检索                         │
        │  SearXNG 网络搜索（可选）                │
        └─────────────────────────────────────────┘
                ↓                   ↓
        [RRF 倒数排名融合]       [网络结果]
                ↓
        [Qianwen Reranker 精排]
                ↓
        [LangChain LCEL 链]
        Prompt 模板 | DeepSeek LLM | StrOutputParser
                ↓
        [SSE 流式输出] → 前端逐 token 渲染
```

不是简单的"检索 + 生成"，而是一条**四层流水线**：语义检索 → 关键词检索 → RRF 融合 → Reranker 精排。每一层都针对一个真实问题设计。

### 技术选型

| 层次 | 技术 | 官方链接 |
|------|------|---------|
| LLM | DeepSeek | [platform.deepseek.com](https://platform.deepseek.com/) |
| Embedding & Reranker | 阿里云千问 | [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com/) |
| 向量库 | FAISS（Meta 开源） | [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss) |
| 关键词检索 | rank-bm25 | [github.com/dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25) |
| 链式框架 | LangChain LCEL | [python.langchain.com](https://python.langchain.com/docs/concepts/lcel/) |
| Web 框架 | Flask | [flask.palletsprojects.com](https://flask.palletsprojects.com/) |
| 网络搜索 | SearXNG（自托管） | [docs.searxng.org](https://docs.searxng.org/) |

---

## 核心技术亮点

### 1. 双路召回 + RRF 融合：解决专有名词漏召回

这是本项目最核心的设计之一。

**问题：** 纯向量检索对专有名词不敏感。搜 "qwen3-rerank 最大输入长度" 可能找不到，但文档里明明有这条内容。

**解法：** [FAISS](https://github.com/facebookresearch/faiss) 语义检索 + [BM25](https://github.com/dorianbrown/rank_bm25) 关键词检索并行跑，通过 **RRF（Reciprocal Rank Fusion）** 合并结果。RRF 最初由 Cormack 等人在 [2009 年的论文](https://dl.acm.org/doi/10.1145/1571941.1572114) 中提出。

RRF 的核心公式只有一行：

```python
score = 1 / (k + rank)   # k=60，rank 从 0 开始
```

两路结果各自按排名计算得分，相加后重排。这个设计有一个优雅的性质：**哪路都排前面的文档，一定是最重要的；只在一路排前面的文档，分数会被拉低但不会被丢弃**。

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

---

### 2. Reranker 精排：解决 Lost in the Middle 问题

双路召回之后，还有几十条候选，不能全部塞给 LLM。

**问题：** 向量相似度排序的是"语义接近"，不是"真正有用"。斯坦福 2023 年的研究（[Lost in the Middle](https://arxiv.org/abs/2307.03172)）表明，LLM 对超长上下文的注意力会集中在开头和结尾，中间的内容容易被忽略。

**解法：** 用 **[阿里云 qwen3-rerank](https://help.aliyun.com/zh/model-studio/developer-reference/rerank-api)（交叉编码器）** 对候选文档重新打分。

- **Embedding（双编码器）**：查询和文档分别编码，余弦相似度比较，快但粗糙
- **Reranker（交叉编码器）**：把查询和文档拼在一起送进模型，直接输出相关性分数，慢但精准

精排之后只保留 Top-3 送给 LLM，上下文质量远超原始检索结果。

---

### 3. 对话引用检测：解决上下文失忆

**问题：** 用户问"你刚才说的那个参数怎么设置"，这是对话引用，检索知识库必然找不到答案。

**解法：** 在检索前先判断问题类型，命中对话引用模式则直接走 [LangChain 对话历史](https://python.langchain.com/docs/concepts/chat_history/)，跳过检索。

```python
_DIALOGUE_REF = re.compile(
    r'(上一个|上一条|刚才|之前(说|问|讲|提到?)?|你说|你提到|'
    r'你刚|继续|你上面|你前面|解释一下刚|你说的)',
    re.IGNORECASE
)

def _is_dialogue_ref(query: str, history: list) -> bool:
    if not history:
        return False
    if _DIALOGUE_REF.search(query):
        return True
    # 短问题（≤15字）+ 指代词 → 大概率是对话引用
    if len(query) <= 15 and re.search(r'(这个|那个|它|这些|那些)', query):
        return True
    return False
```

---

### 4. SSE 流式输出 + 开头套话过滤

**问题：** 模型输出总喜欢以"根据您提供的知识库内容……"开头，影响用户体验。

**解法：** 基于 [SSE（Server-Sent Events）](https://developer.mozilla.org/zh-CN/docs/Web/API/Server-sent_events/Using_server-sent_events) 协议流式输出，同时在缓冲区攒 40 个字符，用正则判断并清理，再推给前端。

```python
_BAD_OPENER = re.compile(
    r'^[\s\n]*(根据(提供的|以上|本|这些)?(知识库|资料|搜索结果|参考)(内容|信息|数据)?'
    r'[，,。]?\s*|知识库(中)?(提到|说明|显示|记录)[，,]?\s*)',
    re.IGNORECASE
)

# 流式输出时，先缓冲 40 字符再决定是否清理开头
BUF_SIZE = 40
if not buf_flushed:
    buf += chunk
    if len(buf) >= BUF_SIZE:
        buf = _BAD_OPENER.sub('', buf, count=1).lstrip()
        buf_flushed = True
        yield _sse({'type': 'token', 'content': buf})
```

---

### 5. 熔断器保护网络搜索

**问题：** 网络搜索偶尔超时，如果每次都等到超时，会严重影响主流程响应速度。

**解法：** 实现 [Circuit Breaker 模式](https://martinfowler.com/bliki/CircuitBreaker.html)——连续失败 3 次后自动熔断 60 秒，期间跳过网络搜索直接用知识库回答。

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
            if self._opened_at and time.time() - self._opened_at > self._timeout:
                self._opened_at = None   # 超时自动恢复
                self._failures = 0
            return self._opened_at is not None
```

网络搜索使用 [SearXNG](https://docs.searxng.org/)（自托管搜索引擎），无 API 限额，隐私保护，一条 Docker 命令即可启动。

---

### 6. 5 种分块策略 + 自动探测

**为什么分块策略这么重要？** 切分质量直接决定召回质量，这是很多人忽视的环节。[LangChain 文档](https://python.langchain.com/docs/concepts/text_splitters/)中提供了基础切分器，本项目在此基础上做了大量针对中文文档结构的扩展。

| 策略 | 适用场景 | 原理 |
|------|---------|------|
| `markdown` | 有 `##`/`###` 标题的技术文档 | 按标题层级保持语义完整性 |
| `regex` | `1.1 xxx` 编号的 Q&A 文档 | 自定义正则匹配分界点 |
| `semantic` | 无结构散文、报告 | 相邻句子 Embedding 相似度骤降处切割（基于 [SemanticChunker](https://python.langchain.com/docs/how_to/semantic-chunker/)） |
| `recursive` | 通用保底 | 按段落→句子→字符递归切割 |
| `auto` | 不确定时 | 自动探测文档结构，选最合适策略 |

每个知识库子目录可以放一个 `kb_config.json` 单独指定策略：

```json
{ "strategy": "markdown", "heading_level": 3, "chunk_size": 1000 }
```

---

## 19 篇配套教学文档

这是项目区别于其他开源 RAG 项目最大的地方。

`lessons/` 目录包含 19 篇系统教学文档，覆盖完整的知识体系：

| 阶段 | 内容 |
|------|------|
| 概念基础 | LLM调用 · 文本切分 · Embedding向量化 · 向量数据库 · 文档加载与解析 |
| 检索核心 | BM25关键词检索 · 混合检索与RRF融合 · Reranker精排 · MultiQuery扩写 · 5种分块策略进阶 |
| 系统集成 | RAG完整流程 · LCEL链式编程 · Prompt模板与提示词工程 · SSE流式输出 · 对话历史管理 |
| 工程实践 | 网络搜索与熔断器 · Flask Web服务设计 · 配置管理与工程实践 |
| 实战调优 | RAG五大痛点与解决方案 |

每篇文档的结构是：**原理讲解 → 代码示例 → 对应主项目位置 → 动手练习题**。

读完 19 节，主项目每一行代码的设计动机你都能说清楚。这对于面试 LLM 应用岗位非常有帮助——说出"为什么用 RRF 而不是直接取 top-K"，比说"我用过 LangChain"含金量高得多。

---

## 系统功能展示

系统提供完整的 Web 界面，支持：

- **流式问答**：检索完成立刻推参考来源，LLM 逐 token 输出，无白屏等待
- **知识库管理**：拖拽上传文档，一键重建向量索引，支持 .txt / .md / .pdf
- **模型热切换**：支持 [DeepSeek](https://platform.deepseek.com/) / [千问](https://dashscope.console.aliyun.com/) / [OpenAI](https://platform.openai.com/) / 本地 [Ollama](https://ollama.com/)，填写配置后测试连接，保存即生效
- **提示词管理**：system.txt 支持存档/回滚，变量插值，API 热重载

---

## 快速上手

**所需 API Key：**
- [DeepSeek 开放平台](https://platform.deepseek.com/) — LLM 对话，价格极低（约 ¥1/百万 token）
- [阿里云 DashScope](https://dashscope.console.aliyun.com/) — Embedding（`text-embedding-v2`）+ Reranker（`qwen3-rerank`）

```bash
# 1. 克隆仓库
git clone https://github.com/Lanqingsong/rag-from-scratch.git
cd rag-from-scratch

# 2. 创建环境并安装依赖（需要 Python 3.10+，推荐 conda）
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY 和 QIANWEN_API_KEY

# 4. 放入你的文档
# 将 .txt / .md / .pdf 文件放入 knowledge_base/ 目录

# 5. 启动
python app.py
# 浏览器访问 http://localhost:5000，首次启动自动构建向量库
```

可选：启用 [SearXNG](https://docs.searxng.org/) 网络搜索（需要 [Docker](https://www.docker.com/)）：

```bash
docker run -d --name searxng -p 8080:8080 searxng/searxng
```

不配置 Docker 也可正常运行，仅禁用网络搜索功能。

---

## 为什么不用 Coze / Dify / FastGPT？

这是被问最多的问题，我的核心理由有三点：

**1. 企业场景常常进不去。** 很多企业内网不允许安装第三方平台，安全审计也不允许 SaaS 工具处理内部文档。本项目纯 Python，能跑在任何 Python 环境里，无需 Docker、无需公网。

**2. 灵活性差距是量级的。** 平台工具是黑盒，你改不了检索逻辑。本项目每个节点都在代码里，想把 BM25 权重从 0.5 调成 0.3，改一行；想在 Reranker 前加关键词过滤，插一个函数；想同时融合本地知识库和实时网络搜索，原生支持。这种可编程性在企业定制场景里至关重要。

**3. 学习价值不同。** 用平台搭出一个 RAG，你只知道 RAG 能做什么。读完本项目的 19 节教学，你会明白每个设计背后的"为什么"——这才是真正理解大模型应用开发的必经之路。

---

## 结语

这个项目是我在做机器视觉知识库系统过程中积累下来的踩坑实录和工程经验，以 [MIT 协议](https://github.com/Lanqingsong/rag-from-scratch/blob/main/LICENSE)全部开源，可自由用于学习、二次开发和商业部署。

如果你也在做 RAG 相关的项目，或者正在学习 [LangChain](https://python.langchain.com/) 和大模型应用开发，希望这个项目对你有帮助。

> 📌 **GitHub：[https://github.com/Lanqingsong/rag-from-scratch](https://github.com/Lanqingsong/rag-from-scratch)**  
> 欢迎点 ⭐ Star · 提 Issue · 提 PR

---

*本项目由 [lanqingsong874953727@outlook.com](mailto:lanqingsong874953727@outlook.com) 与 AI 助手协作开发。*
