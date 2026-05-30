<div align="right">
  <strong>中文</strong> | <a href="README_EN.md">English</a>
</div>

<div align="center">

# RAG 知识库问答系统

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1c3c5e)](https://langchain.com)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

**把自己的文档放进去，5 分钟搭一个私有知识库问答系统。**  
同时也是一套完整的 RAG 学习资料——19 篇教学文档，覆盖从 Embedding 原理到工程调优的全流程，每一行代码为什么这样写都有解释。

---

## 它能做什么

**① 开箱即用的知识库**：上传 `.md` / `.txt` / `.pdf` 文档，系统自动切分、向量化、建索引，支持流式问答、多模型切换、多轮对话，界面直接可用。

**② 比 Dify / Coze 更灵活的私有部署**：完全本地运行，数据不出服务器。文档切分策略可控（5 种，按目录单独配置）、检索管道可定制（向量 + 关键词 + 精排三层可调）、代码全部开放，遇到 Dify 满足不了的需求直接改代码。

**③ 适合深入学习 RAG 的工程实现**：如果你想搞清楚向量检索为什么对型号编码不敏感、Reranker 和 Embedding 的区别是什么、为什么对话历史要从最新往前截而不是从最早往后保留——这 19 篇文档比大多数教程讲得更深。

---

## 界面预览

![主界面](docs/screenshots/ScreenShot_2026-05-28_115847_456.png)

<table>
<tr>
<td width="50%">

**知识库管理** — 拖拽上传，一键重建索引

![知识库管理](docs/screenshots/ScreenShot_2026-05-28_115901_013.png)

</td>
<td width="50%">

**模型配置** — 在线切换服务商，测试后即时生效

![模型配置](docs/screenshots/ScreenShot_2026-05-28_115911_567.png)

</td>
</tr>
</table>

---

## 快速上手

**需要准备：** Python 3.10+、[DeepSeek API Key](https://platform.deepseek.com/)、[阿里云 DashScope API Key](https://dashscope.console.aliyun.com/)（Embedding + Reranker）

```bash
git clone https://github.com/Lanqingsong/rag-from-scratch.git
cd rag-from-scratch

conda create -n rag_env python=3.10 -y && conda activate rag_env
pip install -r requirements.txt

cp .env.example .env
# 编辑 .env，填写 DEEPSEEK_API_KEY 和 QIANWEN_API_KEY

# 把文档放入 knowledge_base/（支持 .md / .txt / .pdf）
python app.py
```

浏览器访问 `http://localhost:5000`，首次启动自动构建向量库。  
不配 SearXNG 时，在 `.env` 里设 `WEB_SEARCH_ENABLED=False`，其余功能完整可用。

---

## 相比 Dify / Coze，这个项目的优势

Dify、Coze、FastGPT 是优秀的产品，但在以下场景它们不够用：

| 场景 | 集成工具 | 本项目 |
|------|---------|--------|
| 文档切分 | 固定字符数，无法按文档结构切 | 5 种策略，Markdown 按标题切、Q&A 按编号切、散文按语义切 |
| 召回型号编码 | 向量检索对低频专有名词不敏感 | BM25 关键词检索兜底，精确字符匹配 |
| 自定义检索逻辑 | 黑盒，无法干预 | 代码开放，每层参数可调，可插入新过滤步骤 |
| 私有部署 | SaaS 或依赖 Docker 镜像 | 纯 Python，`python app.py` 一行启动 |
| 知识库 + 网络搜索联动 | 逻辑固定 | 两路并行检索，融合策略可自定义 |

---

## 检索架构

系统内置四层检索管道，每层解决一个具体问题：

```
FAISS 向量检索      → 语义理解，同义词、上下文关联
  +
BM25 关键词检索     → 精确命中型号、参数名、低频专有名词
  ↓
RRF 融合排名        → 合并两路结果（score = 1 / (60 + rank)）
  ↓
Qianwen Reranker   → 交叉编码器精排，只把最相关的 3 条送给 LLM
```

此外，系统在进入检索前会先做**对话引用检测**：识别"你刚才说的"、"继续"这类问题，直接走对话历史而不触发检索，避免无意义的向量搜索。

---

## 文档切分

切分方式决定召回质量，比 chunk_size 更重要。通过每个子目录的 `kb_config.json` 单独配置：

| 策略 | 适用文档 | 示例配置 |
|------|---------|---------|
| `markdown` | 有 `##`/`###` 标题的手册 | `{"strategy":"markdown","heading_level":3}` |
| `regex` | 编号格式文档（`1.1 xxx`） | `{"strategy":"regex","pattern":"\\d+\\.\\d+\\s+[^\\n]+"}` |
| `semantic` | 无结构散文 | `{"strategy":"semantic","threshold_value":95}` |
| `recursive` | 通用保底 | `{"strategy":"recursive"}` |
| `auto` | 不确定时自动探测结构 | `{"strategy":"auto"}` ← 默认 |

---

## 学习路径（19 篇教学文档）

`lessons/` 目录是这套系统的配套教材，从调一个 LLM 到调优一个完整 RAG 系统，全程对照主项目代码：

| 阶段 | 文档 |
|------|------|
| 概念基础 | [01 LLM调用](lessons/01_LLM调用基础.md) · [02 文本切分](lessons/02_文本切分策略.md) · [03 Embedding](lessons/03_Embedding向量化.md) · [04 向量数据库](lessons/04_向量数据库.md) · [05 文档加载](lessons/05_文档加载与解析.md) |
| 检索核心 | [06 BM25](lessons/06_BM25关键词检索.md) · [07 混合检索RRF](lessons/07_混合检索与RRF融合.md) · [08 Reranker](lessons/08_Reranker精排.md) · [09 MultiQuery](lessons/09_MultiQuery检索扩写.md) · [15 分块策略进阶](lessons/15_5种分块策略进阶.md) |
| 系统集成 | [10 RAG完整流程](lessons/10_RAG完整流程.md) · [11 LCEL链](lessons/11_LCEL链式编程.md) · [12 Prompt工程](lessons/12_Prompt模板与提示词工程.md) · [13 SSE流式](lessons/13_SSE流式输出.md) · [14 对话历史](lessons/14_对话历史管理.md) |
| 工程实践 | [16 熔断器](lessons/16_网络搜索与熔断器.md) · [17 Flask服务](lessons/17_Flask_Web服务设计.md) · [18 配置管理](lessons/18_配置管理与工程实践.md) |
| 实战调优 | [19 RAG五大痛点与解决方案](lessons/19_RAG五大痛点与解决方案.md) |

详见 [lessons/README.md](lessons/README.md)

---

## 项目结构

```
rag-from-scratch/
├── app.py               # Flask 服务：路由、SSE、知识库管理 API
├── knowledge_base.py    # FAISS + BM25 + RRF + Reranker + MultiQueryRetriever
├── splitters.py         # 5 种分块策略 + Auto 自动探测
├── llm_client.py        # LCEL 链，4 种 Prompt 模式自动选择
├── prompts.py           # Prompt 热重载 + 变量插值
├── web_search.py        # SearXNG 网络搜索 + 熔断器
├── config.py            # 三层配置（.env → model_config.json → 默认值）
│
├── knowledge_base/      # 放知识文档（.md / .txt / .pdf）
├── vector_store/        # FAISS 向量库（自动生成）
├── prompt_templates/    # system.txt + variables.json + 版本存档
│
└── lessons/             # 19 篇教学文档 + 配套实验代码
```

---

## 配置参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TOP_K_RESULTS` | `5` | 检索候选文档数 |
| `RERANK_TOP_K` | `3` | Reranker 后保留数（送入 LLM） |
| `KB_RELEVANCE_SCORE` | `0.62` | FAISS 相似度阈值，召回 0 条时调低 |
| `MULTI_QUERY_RETRIEVAL` | `True` | 启用多角度查询改写 |
| `MAX_HISTORY_TOKENS` | `3000` | 对话历史 Token 预算 |
| `WEB_SEARCH_ENABLED` | `True` | 启用网络搜索（需 SearXNG） |

切换模型：点击界面"模型配置"，或直接编辑 `model_config.json`：

| 服务商 | base_url | 示例模型 |
|--------|----------|---------|
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` |
| 阿里云 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-max-latest` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| 本地 Ollama | `http://localhost:11434/v1` | `qwen2.5:14b` |

---

## 常见问题

**Q: 启动报 `DEEPSEEK_API_KEY 未设置`**  
项目根目录需要有 `.env` 文件，内容 `DEEPSEEK_API_KEY=sk-xxx`。

**Q: 检索命中 0 条（日志 `FAISS=0 BM25=0`）**  
先确认 `vector_store/` 已生成。仍为 0 则把 `KB_RELEVANCE_SCORE` 调低到 `0.5`。

**Q: 换成自己领域的知识库**  
替换 `knowledge_base/` 里的文档 → 修改 `prompt_templates/system.txt` 的角色设定 → 删除 `vector_store/` → 重启自动重建。

**Q: 启用网络搜索（SearXNG）**  
```bash
docker run -d --name searxng -p 8080:8080 searxng/searxng
docker exec searxng sed -i 's/- html/- html\n  - json/' /etc/searxng/settings.yml
docker restart searxng
```

---

## 关于作者

大家好，我是 **lanqingsong**，一名热衷于 AI 工程实践的开发者。

这个项目诞生于我系统学习 RAG 技术的过程中——一边啃论文和源码，一边把踩过的坑和总结的最佳实践沉淀成文档和可运行代码。目标是让下一个入门 RAG 的人不必走一样的弯路。

如果你也在做类似的事情，欢迎互相交流；如果你觉得这个项目有价值，欢迎 Star 或贡献 PR。

---

## 开源协议

[MIT License](LICENSE)

---

## 欢迎交流 🙌

这个项目是我在学习和实践 RAG 过程中沉淀下来的，如果你在使用中遇到任何问题、有改进建议，或者想一起探讨 RAG 相关技术，都非常欢迎！

**几种联系方式：**

- 🐛 **发现 Bug / 有功能建议** → [提交 Issue](../../issues)，描述越详细越好
- 💬 **技术讨论 / 使用问题** → 直接在 Issue 里提问，我会尽量回复
- 📧 **其他合作** → [lanqingsong8749@outlook.com](mailto:lanqingsong8749@outlook.com) · [874953727@qq.com](mailto:874953727@qq.com)

不管是初学者的入门困惑，还是高手的架构建议，都欢迎来聊。能帮到你就是最好的回报。

---

<div align="center">

如果本项目对你有帮助，欢迎点个 ⭐

[教学文档](lessons/README.md) · [提交 Issue](../../issues) · [发起讨论](../../discussions)

</div>
