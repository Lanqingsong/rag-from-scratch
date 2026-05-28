# 从生产踩坑到开源：一个真实 RAG 系统的工程决策记录

> **GitHub**：[https://github.com/Lanqingsong/rag-from-scratch](https://github.com/Lanqingsong/rag-from-scratch)  
> **技术栈**：[LangChain](https://python.langchain.com/) · [DeepSeek](https://platform.deepseek.com/) · [FAISS](https://github.com/facebookresearch/faiss) · [rank-bm25](https://github.com/dorianbrown/rank_bm25) · [阿里云 DashScope](https://dashscope.console.aliyun.com/) · [Flask](https://flask.palletsprojects.com/) · [SearXNG](https://docs.searxng.org/)  
> **开源协议**：MIT

---

## 动机：一个系统上线之后才开始的故事

这个项目的起点不是"我想学 RAG"，而是一个已经上线、已经在被用户使用的系统开始暴露问题。

背景是机器视觉领域的知识库问答——把大量技术手册、选型文档、工艺参数表整合进去，让工程师可以直接问"这款相机的像素尺寸是多少""镜头焦距怎么算"之类的问题。早期版本的逻辑很简单：用千问 Embedding 把文档向量化，存进 FAISS，用户提问时检索 top-5，塞进 Prompt，让 DeepSeek 回答。

Demo 阶段效果挺好，领导也满意。但真正让工程师用起来之后，问题一个接一个浮出来。

每一个问题背后，我都走了一段弯路才找到根本原因。这篇文章想把这个过程原原本本写下来——不是为了展示最终架构有多漂亮，而是为了记录那些弯路和推理过程，因为我认为这比结论更有价值。

---

## 第一个问题：向量检索的专有名词盲区

### 现象

系统上线后不久，一位工程师反映：他问"CS-040GX 的最大帧率是多少"，系统回答说"没有找到相关资料"。但我知道文档里明明有这个型号的参数表。

手动翻了一遍日志，发现 FAISS 检索返回了 0 条结果。

### 根本原因

这让我开始重新理解 Embedding 模型在做什么。

Embedding 模型的训练目标是让语义相近的句子在向量空间里距离接近。"相机分辨率"和"图像像素数量"会被映射到相近的位置，因为它们在大量语料里经常共现、互相解释。

但"CS-040GX"这个字符串在训练语料里出现频率极低，模型对它几乎没有语义记忆。当用户输入这个型号时，它的向量表示基本上就是个随机方向，和文档里那段包含参数表的向量距离很远，余弦相似度低于阈值，被过滤掉了。

**这不是 Bug，这是 Embedding 的设计特性**。它擅长的是语义理解，不是精确字符匹配。

### 解决方案：BM25 作为互补路

BM25（Best Match 25）是 TF-IDF 的改进版本，它的核心逻辑是：一个词在文档里出现越多、在全库里越罕见，这个文档对该词的相关性越高。用公式表达：

```
score(D, Q) = Σ IDF(qᵢ) · (f(qᵢ,D) · (k₁+1)) / (f(qᵢ,D) + k₁ · (1 - b + b · |D|/avgdl))
```

其中 `f(qᵢ,D)` 是词频，`k₁` 控制词频饱和（通常取 1.2~2.0），`b` 控制文档长度归一化（通常取 0.75）。

对"CS-040GX"这个查询，BM25 会精确找到包含这个字符串的文档片段，不管它的语义是什么。

**两路并行运行，结果通过 RRF 融合**：

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

RRF 公式 `1/(k+rank)` 中，`k=60` 这个默认值来自 [Cormack 等人 2009 年的论文](https://dl.acm.org/doi/10.1145/1571941.1572114)——他们在实验中发现 k=60 在多种任务上表现稳健，原因是它对排名靠前的文档给予适度奖励，同时不会过度惩罚排在后面的文档。

加了 BM25 之后，型号查询的命中率从不到 30% 提升到了 90% 以上。

---

## 第二个问题：检索到了，LLM 却回答错了

### 现象

BM25 上线后，召回率好多了，但出现了新问题：有时候我能在参考资料里看到正确答案，但 LLM 给出的回答却是错的或者不完整的。

刚开始我以为是 Prompt 写得不够好，改了好几版，没什么改善。

### 根本原因

后来读到了斯坦福 2023 年的一篇研究论文：[Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)。

论文里的实验结论让我印象深刻：当把多个文档片段拼接成上下文喂给 LLM 时，LLM 对开头和结尾部分的利用率远高于中间部分。如果把正确答案放在第 1 条或最后一条，准确率约 70%；如果放在第 6~9 条（中间），准确率降至约 40%。

**问题不是 Prompt，是上下文的排列顺序和数量**。我把 top-10 全塞进去，正确答案可能恰好排在第 5 位。

### 解决方案：Reranker 作为第二层过滤

Embedding（双编码器）和 Reranker（交叉编码器）的核心区别是：

- **双编码器**：查询和文档各自独立编码成向量，然后计算余弦相似度。速度快，但因为编码时两者互不可见，精度受限。
- **交叉编码器**：把查询和文档**拼接在一起**送进同一个模型，模型在每一个 attention 层都能看到两者的交互，直接输出一个相关性分数。慢，但精度远高于双编码器。

Reranker 适合做"最后一公里"的精选：召回阶段用快速的 Embedding 拿到几十个候选，然后用 Reranker 精确打分，只保留 top-3 送给 LLM。

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

加了 Reranker 之后，给 LLM 的上下文从 10 条压缩到 3 条，而且这 3 条是被交叉编码器仔细审查过的。主观体感上，回答的准确性和完整性都有明显提升。

**这里有个值得注意的权衡**：Reranker 的 API 调用需要额外的网络请求和费用。如果 top-K 本来就很准（比如文档质量很高、用户问题很精确），Reranker 的提升有限，但成本始终存在。我设置的阈值是：当召回文档数超过 3 条时才调用 Reranker，3 条以内直接用。

---

## 第三个问题：对话引用导致的语义失联

### 现象

用户问完"镜头焦距怎么算"，紧接着问"那工作距离呢"，或者"你上面说的那个公式能不能展开解释一下"。

系统对第二个问题的处理逻辑是：去知识库检索"那工作距离呢"和"你上面说的那个公式"——当然什么都找不到，然后回答"没有找到相关资料"或者生成了一段完全无关的内容。

### 根本原因

这类问题的语义依赖于对话历史，它们是**指代性问题**——"那个"、"上面说的"、"刚才"指向的是对话上下文里的某个概念，而不是知识库里的某个文档。

知识库检索的假设是：用户的问题包含足够的信息来定位文档。这个假设在单轮问答里成立，在多轮对话里经常失效。

### 解决方案：检索之前先判断问题类型

在进入检索流程之前，先做一个轻量级的意图判断：

```python
_DIALOGUE_REF = re.compile(
    r'(上一个|上一条|刚才|之前(说|问|讲|提到?)?|你说|你提到|'
    r'你刚|继续|你上面|你前面|解释一下刚|你说的)',
    re.IGNORECASE
)

def _is_dialogue_ref(query: str, history: list) -> bool:
    if not history:          # 没有历史就不可能是对话引用
        return False
    if _DIALOGUE_REF.search(query):
        return True
    if len(query) <= 15 and re.search(r'(这个|那个|它|这些|那些)', query):
        return True          # 短问题 + 指代词，高概率引用历史
    return False
```

命中时，跳过知识库检索，直接把完整的对话历史连同问题送给 LLM。

**这个方案的局限性**：正则匹配依赖关键词，会有漏检（用户说"继续"但实际是全新话题）和误检（用户说"你说的那个文档里有没有这个"，其实是在问知识库）。更健壮的方案是用一个小模型做意图分类，但这会增加每次请求的额外 LLM 调用开销，对于当前体量来说得不偿失。我选择接受少量错误，在日志里记录下来，后续根据频率决定是否值得优化。

另一个值得记录的决策是对话历史的 Token 预算管理。对话历史不能无限累积，否则会撑爆上下文窗口。这里的策略是**从最新一轮往前截取**，而不是从最早的往前保留：

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

原因：距离当前问题最近的对话轮次，对理解当前问题的价值最高。当历史过长时，优先丢弃最早的轮次。

---

## 第四个问题：流式输出的开头套话

### 现象

这个问题比较"表面"，但用户反应强烈：系统每次回答几乎都以"根据您提供的知识库内容"或"根据以上参考资料"开头，而且这句话会在流式输出里第一个出现，用户等了一两秒看到这句话，有一种被敷衍的感觉。

### 根本原因

这是大模型经过 RLHF 训练之后形成的习惯性输出模式。在人工标注的训练数据里，引用来源被认为是"诚实、负责任"的表现，因此这类开头被反复正强化。模型学会了：当有参考资料时，先说"根据资料"是个好习惯。

改 Prompt 能有效果，但不能根治——不同模型、不同温度下这个行为的强度不同，有时候 Prompt 管一段时间后效果又退化。

### 解决方案：在流式缓冲区里拦截

SSE（[Server-Sent Events](https://developer.mozilla.org/zh-CN/docs/Web/API/Server-sent_events)）流式输出是逐 token 发送的，如果每个 token 到了就立刻推给前端，就没有机会做开头过滤。

解决方法是在服务端设置一个缓冲区，攒够 40 个字符再决定是否清理开头，然后推出去：

```python
_BAD_OPENER = re.compile(
    r'^[\s\n]*(根据(提供的|以上|本|这些)?(知识库|资料|搜索结果|参考)(内容|信息|数据)?'
    r'[，,。]?\s*|知识库(中)?(提到|说明|显示|记录)[，,]?\s*)',
    re.IGNORECASE
)

buf = ""
buf_flushed = False
BUF_SIZE = 40

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
```

40 个字符这个数字是经验值——中文"根据您提供的知识库内容，"大约 15 个字符，留 40 个字符的缓冲足以判断但又不会让用户感觉到明显的延迟。

这个方案的副作用：当 LLM 的回答本身很短（比如只有 30 个字），缓冲区在回答结束后才一次性 flush，用户体验是"等了一下，内容一次性出来"，而不是逐字流式。项目里对这个 edge case 做了补偿处理：

```python
# 回答结束后，剩余缓冲一次性推出
if buf:
    buf = _clean(buf)
    yield _sse({'type': 'token', 'content': buf})
```

---

## 第五个问题：Prompt 工程的版本混乱

### 现象

这个问题是在某次重大改版之后才意识到的。我为了适配一个新的提问场景改了 `system.txt`，改完之后发现之前某个问题类型的回答质量下降了，想回到上一版本，但没有任何记录——只有一个一直被覆盖的文件。

### 根本原因

Prompt 工程在实践中是一个反复迭代的过程，远比写代码更频繁地修改。但它通常不被当作"代码"来对待，没有版本控制，没有回滚机制。

这导致了一个常见的困境：改好了某类问题，却搞坏了另一类；改回去，之前的进步又没了。没有 A/B 对比的能力，优化就是在瞎撞。

### 解决方案：提示词版本管理 + 变量系统

系统里实现了三层提示词管理：

**第一层：版本存档 API**

```bash
# 存档当前版本
POST /api/prompts/save_version
# 返回 {"version": "system_20260528_143022.txt"}

# 激活历史版本
POST /api/prompts/activate/system_20260528_143022.txt
# 立即生效，无需重启
```

**第二层：变量插值**

`system.txt` 里可以写占位符：

```
你是 {{domain}} 领域的助手，服务对象是 {{user_type}}。
```

`variables.json` 里存值：

```json
{
  "system_vars": {
    "domain": "机器视觉",
    "user_type": "工程师"
  }
}
```

这样调整定制内容不需要改主文件，也减少了版本混乱的可能性。

**第三层：热重载**

```bash
POST /api/prompts/reload
```

修改 `system.txt` 或 `variables.json` 后调用这个接口，不需要重启 Flask 进程，对正在进行的对话无影响。

---

## 一些架构层面的思考

### 为什么用 FAISS 而不是 ChromaDB

ChromaDB 有更好的开箱即用体验，有 Web UI，有持久化。但在这个场景里，FAISS 有两个决定性优势：

1. **速度**：FAISS 是纯 C++ 实现，Python 绑定开销极小，在几百到几千个向量的规模下，检索延迟在毫秒级。ChromaDB 有额外的 HTTP 层或进程间通信开销。
2. **无依赖**：FAISS 是一个文件库，向量库就是本地的两个文件（`.faiss` 和 `.pkl`），整个知识库可以随 ZIP 打包带走，部署到任何地方。

**代价**：没有 UI，没有在线更新（需要重建），没有元数据过滤。在当前场景下这些都不是问题。

### 为什么用 Flask 而不是 FastAPI

FastAPI 在异步性能和 API 文档上确实更好。选 Flask 的原因是：

1. **SSE 流式输出**：Flask 的 `Response(generator, mimetype='text/event-stream')` 模式非常简洁，而 FastAPI 的 SSE 实现需要额外的 `sse-starlette` 库，还需要处理异步生成器的兼容性问题。
2. **现有代码库**：项目里很多模块是同步的（FAISS 检索、BM25 检索），用异步框架反而需要用 `run_in_executor` 来包装，不如直接用同步框架来得干净。

如果未来需要支持高并发，切到 FastAPI + 异步是合理的升级路径。

### 为什么用 LangChain 而不是自己写

LangChain 的争议很大——过度抽象、接口不稳定、版本间 breaking change 很多。但在这个项目里，它解决了几个真实问题：

1. **LCEL（LangChain Expression Language）的管道语法**：`Prompt | LLM | Parser` 这种写法让链的组合和替换变得极其简洁，切换 LLM 提供商只需要换一个对象。
2. **MultiQueryRetriever**：把用户的单个问题改写成 3 个不同角度的查询，覆盖用户可能的表述变体。这在没有 LangChain 的情况下也能实现，但代码量会多不少。
3. **SemanticChunker**：基于 Embedding 相似度的语义切分，LangChain 已经有成熟实现，不必从头写。

接受 LangChain 的前提是：理解它底层在做什么，不依赖它的高层封装来做不理解的事情。

---

## 开源与教学的初衷

`lessons/` 目录里的 19 篇文档是这个项目里花时间最多的部分。

写这些文档的原因很直接：我在做这个系统的时候，走了很多弯路，很多时候是因为找不到把"原理"和"工程实现"连接起来的资料——要么是只有公式没有代码，要么是只有代码没有解释，要么是教程基于一个跟我实际场景完全不同的假设。

我希望这 19 篇文档能做到：看完之后，这个项目每一行代码的设计动机你都能说清楚，而不只是"这是 copy 来的"。

如果你正在做类似的项目，或者正在学习 LangChain 和大模型应用开发，欢迎参考。如果你发现了问题或者有更好的方案，欢迎提 Issue 或 PR。

> 📌 **GitHub：[https://github.com/Lanqingsong/rag-from-scratch](https://github.com/Lanqingsong/rag-from-scratch)**

---

*本项目由 [lanqingsong874953727@outlook.com](mailto:lanqingsong874953727@outlook.com) 与 AI 助手协作开发。*

*编写时间：2026 年 5 月 28 日*
