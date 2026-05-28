# RAG 系统学习路径

本目录是从零开始构建 RAG（检索增强生成）系统的完整学习路径。  
每一节包含一个 **概念讲解文档**（.md）和对应的 **可运行代码**（.py）。

学完全部内容后，你将理解主项目 `knowledge_base.py` / `app.py` 中每一行代码的设计动机。

---

## 学习顺序

| 序号 | 文档 | 配套代码 | 核心概念 |
|------|------|---------|---------|
| 01 | [LLM 调用基础](01_LLM调用基础.md) | `01call_llm.py` | ChatOpenAI、invoke/stream、消息格式 |
| 02 | [文本切分策略](02_文本切分策略.md) | `1~3.py` | chunk_size、overlap、递归切分 |
| 03 | [Embedding 向量化](03_Embedding向量化.md) | `4.向量和Embeddings.py` | 高维向量、余弦相似度、批处理 |
| 04 | [向量数据库](04_向量数据库.md) | `5.chromdb使用.py` | ChromaDB、FAISS、ANN 索引 |
| 05 | [文档加载与解析](05_文档加载与解析.md) | `6.文档读取doc.py` | PDF/DOCX 提取、噪声清洗、元数据 |
| 06 | [BM25 关键词检索](06_BM25关键词检索.md) | *(knowledge_base.py)* | TF-IDF→BM25、专有名词命中 |
| 07 | [混合检索与 RRF 融合](07_混合检索与RRF融合.md) | *(knowledge_base.py)* | 双路召回互补、RRF 公式推导 |
| 08 | [Reranker 精排](08_Reranker精排.md) | *(knowledge_base.py)* | 双编码器 vs 交叉编码器、两阶段检索 |
| 09 | [MultiQuery 检索扩写](09_MultiQuery检索扩写.md) | *(knowledge_base.py)* | LLM 改写查询、召回扩展 |
| 10 | [RAG 完整流程](10_RAG完整流程.md) | *(全部文件)* | 端到端串联、设计决策解析 |
| 11 | [LCEL 链式编程](11_LCEL链式编程.md) | *(llm_client.py)* | `\|` 管道语法、4 种链、热切换 |
| 12 | [Prompt 模板与提示词工程](12_Prompt模板与提示词工程.md) | *(prompts.py)* | 模板设计、变量插值、热重载、版本管理 |
| 13 | [SSE 流式输出](13_SSE流式输出.md) | *(app.py)* | SSE 协议、事件类型、套话过滤、LaTeX 渲染 |
| 14 | [对话历史管理](14_对话历史管理.md) | *(app.py)* | Token 预算、历史截取、HumanMessage |
| 15 | [5 种分块策略进阶](15_5种分块策略进阶.md) | *(splitters.py)* | Semantic/Markdown/Regex/Auto、kb_config |
| 16 | [网络搜索与熔断器](16_网络搜索与熔断器.md) | *(web_search.py)* | SearXNG、Circuit Breaker、指数退避 |
| 17 | [Flask Web 服务设计](17_Flask_Web服务设计.md) | *(app.py)* | RESTful API、安全校验、异步重建 |
| 18 | [配置管理与工程实践](18_配置管理与工程实践.md) | *(config.py)* | 三层级配置、Key 脱敏、优雅降级 |
| 19 | [RAG 五大痛点与解决方案](19_RAG五大痛点与解决方案.md) | *(全部文件)* | 召回失败、LLM 忽略上下文、套话过滤、上下文失忆、Prompt 迭代工程 |

---

## 学习建议

**第一阶段（概念基础）：** 01 → 02 → 03 → 04 → 05  
动手运行每个 `.py` 文件，观察输出，修改参数体验变化。

**第二阶段（检索核心）：** 06 → 07 → 08 → 09 → 15  
这些是本项目最有技术含量的部分，边读文档边对照 `knowledge_base.py` 代码理解。

**第三阶段（系统集成）：** 10 → 11 → 12 → 13 → 14  
理解各组件如何被 LCEL 链和 Flask 服务串联起来。

**第四阶段（工程实践）：** 16 → 17 → 18  
学习生产级代码的容错、安全和配置管理设计。

**第五阶段（实战调优）：** 19  
系统能跑之后，这节讲真实项目中最常踩的 5 类坑及根治方案，是前 18 节知识的综合运用。

---

## 环境准备

```bash
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r ../requirements.txt
```

在 `lessons/` 目录下创建 `.env`（或在根目录配置），填写 API Key：

```env
DASHSCOPE_API_KEY=sk-xxxx      # 千问 Embedding 用
DEEPSEEK_API_KEY=sk-xxxx       # DeepSeek LLM 用
```
