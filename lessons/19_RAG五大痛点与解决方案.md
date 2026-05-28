# 19 · RAG 五大痛点与解决方案

> 这篇文档来自真实踩坑经历。  
> 搭好 RAG 系统后，你大概率会先后遇到这五个问题——它们本质上都有解，但每个坑都需要从对的方向入手。

---

## 痛点一：召回失败——明明有这段内容，就是找不到

### 现象

给系统提问，问题的措辞和知识库里的某个标题**几乎一样**，但系统就是检索不到。  
或者：一段话放在知识库里，原文复制粘贴去问，也不能命中。

### 根因分析

**原因 A：chunk 边界把标题和正文切断了**

切分后的 chunk 只剩正文，标题被丢掉了。用户问"2.1 镜头焦距计算"，  
向量库里存的是"焦距 f = WD × sensor_size / FOV..."，没有"2.1"这几个字，BM25 自然找不到。

```
原文：
  ### 2.1 镜头焦距计算
  焦距 f = WD × sensor_size / FOV...

切分后 chunk 的 page_content：
  "焦距 f = WD × sensor_size / FOV..."   ← 标题丢了！
  metadata: {section_title: "2.1 镜头焦距计算"}  ← 只在元数据里
```

**原因 B：相似度阈值设太高**

`KB_RELEVANCE_SCORE = 0.62` 在某些 Embedding 模型下可能太高，相关文档被过滤掉了。

**原因 C：chunk_size 太小，语义被碎片化**

一个知识点被切成 3 个 chunk，每个 chunk 单独看相关度都不高，全部低于阈值被丢弃。

**原因 D：Embedding 模型对该领域词汇不敏感**

通用 Embedding 对中文专业术语的向量表征质量参差不齐。

### 解决方案

**方案 1：把标题写进 chunk 内容（最直接有效）**

在 `splitters.py` 的 `MarkdownHeadingSplitter._split()` 中，  
生成 chunk 时把标题拼入正文，而不只放元数据：

```python
# 修改前：标题只放 metadata
meta = {**metadata, 'section_title': heading}
chunks.append(Document(page_content=body, metadata=meta))

# 修改后：标题也写进 page_content
body_with_title = f"{heading}\n{body}"
chunks.append(Document(page_content=body_with_title, metadata=meta))
```

这样 BM25 关键词检索也能命中标题词汇，FAISS 向量也包含了标题语义。

**方案 2：降低相似度阈值**

```python
# config.py
KB_RELEVANCE_SCORE = 0.50   # 从 0.62 降到 0.50，先看能不能召回
```

调低后如果出现不相关内容进入答案，再通过 Reranker 精排来过滤。

**方案 3：增大 chunk_size 或调整策略**

```json
// knowledge_base/kb_config.json
{
  "strategy": "markdown",
  "heading_level": 3,
  "chunk_size": 1500    // 从 1000 增到 1500，减少碎片化
}
```

**方案 4：诊断工具——打印完整检索过程**

在 `knowledge_base.py` 的 `search()` 中加日志：

```python
# 临时诊断：关掉阈值过滤，看原始召回了什么
docs_with_score = self.vector_store.similarity_search_with_score(query, k=10)
for doc, score in docs_with_score:
    print(f"score={score:.3f} | {doc.page_content[:60]}")
# 如果目标文档出现但 score < 阈值 → 调低阈值
# 如果目标文档根本不在 Top10 → 问题在分块或 Embedding
```

---

## 痛点二：检索到了，LLM 却视而不见，照样胡说

### 现象

参考资料面板里明确显示了正确答案的来源文档，但 LLM 的回答完全不基于这段内容，  
给出了错误的或凭空捏造的答案。

### 根因分析

**原因 A：Prompt 结构问题——上下文注入位置错误**

把参考资料注入到 `human` 消息而不是 `system` 消息，LLM 对 human 消息里的长文本遵循度较低：

```python
# 错误写法：参考资料放在 human 消息里
ChatPromptTemplate.from_messages([
    ("system", "你是AI助手"),
    ("human", "参考资料：{context}\n\n问题：{query}")  # ← 容易被忽略
])

# 正确写法：参考资料放在 system 消息里
ChatPromptTemplate.from_messages([
    ("system", "你是AI助手\n\n## 参考资料\n{context}"),  # ← 遵循度高
    ("human", "{query}")
])
```

**原因 B：Prompt 指令太弱**

"参考以上内容回答"是弱指令，模型的训练偏好和参考资料冲突时会忽略参考资料。

**原因 C：context 窗口里参考资料被淹没**

对话历史太长 + 参考资料较多，总 Token 超出模型注意力有效范围，  
靠后的参考资料被模型"遗忘"（Lost in the Middle 问题）。

### 解决方案

**方案 1：加强 Prompt 中的指令强度**

```
# prompt_templates/system.txt 修改建议

你是机器视觉领域的资深工程师AI助手。

## 使用参考资料的方式（重要）
你的回答必须以"参考资料"部分的内容为事实依据。
- 若参考资料中有相关内容：基于它推理、整合、深化，不得与它矛盾
- 若参考资料中没有相关内容：明确告知"当前知识库中没有该信息"，不要编造
- 禁止输出任何与参考资料相矛盾的内容

## 回答格式
...
```

**方案 2：在参考资料注入时加结构标记**

```python
# llm_client.py 中格式化 context
def _fmt_kb(self, docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get('section_title', '')
        parts.append(f"【资料{i}】{title}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
```

编号和分隔线让模型更清楚地区分每条资料的边界。

**方案 3：减少送入 LLM 的 context 条数**

```python
# config.py
RERANK_TOP_K = 2   # 从 3 降到 2，减少 Lost in the Middle 风险
```

精而不滥，2 条高质量资料比 5 条中等质量资料效果好。

---

## 痛点三：回答死板——原文照搬，没有思考和整合

### 现象

问一个问题，模型把知识库原文直接复制过来，没有整合、没有推理、没有针对问题的重点提炼。  
加了网络搜索结果之后也没改善，两份资料各自"粘贴"，而不是融合成一个有逻辑的回答。

### 根因分析

Prompt 指令的措辞造成了行为误导。常见的错误指令：

```
❌ "根据以下资料回答问题"   → 模型理解为"摘抄资料"
❌ "仅依据资料作答"         → 模型理解为"不加工，原文输出"
❌ "忠实于原文"             → 直接鼓励了照搬行为
```

### 解决方案

**核心原则：告诉模型资料是"原材料"，回答是"成品"**

```
# system.txt 关键段落的正确写法

## 如何使用参考资料
参考资料是你推理的原材料，不是答案模板。

正确流程：
1. 理解资料中的核心信息和逻辑
2. 结合问题，判断哪些内容真正相关
3. 用自己的语言重新组织，补充资料以外你知道的相关专业知识
4. 给出系统、有结构的回答（标题 + 列表 + 公式）

错误行为（绝对禁止）：
- 原文复制粘贴
- 以"根据资料/知识库"开头
- 资料有什么就写什么，不针对问题做取舍
```

**区分"知识库资料"和"网络搜索资料"的处理方式**

本项目已有 `DUAL_SOURCE` Prompt 模式，把本地资料和网络资讯分开标注，  
让模型能有意识地对两类来源做不同处理：

```python
# prompts.py 中的 DUAL_SOURCE 模板
DUAL_SOURCE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + "\n\n## 本地知识库资料（权威，以此为准）\n{kb_context}"
               "\n\n## 互联网最新资讯（补充参考）\n{web_context}"),
    ...
])
```

在 system.txt 中对这两类来源的使用方式做差异化说明：

```
## 两类资料的使用优先级
本地知识库资料：权威来源，优先采信，回答时以此为核心依据
互联网资讯：补充参考，用于填补本地资料的空白或更新时效性信息
两者有冲突时：以本地知识库为准，同时说明互联网有不同说法
```

---

## 痛点四：上下文失忆——加了对话历史，反而更蠢了

### 现象

- 用户说"刚才你提到的那个公式"，系统去知识库搜"公式"，完全无视上下文
- 多轮对话后，模型开始混淆不同话题的内容
- 每轮对话都触发知识库检索，即使问的是"嗯"或"继续"

### 根因分析

**原因 A：对话引用检测覆盖不全**

现有的 `_is_dialogue_ref()` 正则只覆盖了常见说法，很多自然的指代表达没有被识别：

```python
# 当前正则漏掉的场景：
"你说的那个公式"     ← 指代词 + 名词
"解释得更详细点"     ← 追问上文
"举个例子"          ← 针对上文的延伸请求
"不对，重新解释"     ← 否定上文要求重来
```

**原因 B：对话历史 Token 过长导致 LLM 注意力稀释**

历史超过 2000 Token 后，早期对话内容的权重急剧下降，模型"忘记"了几轮前说的话，  
只能重新去知识库搜。

**原因 C：检索路由缺失——没有"是否需要检索"的判断**

每个问题都无脑去检索知识库，即使问题明显是对上文的追问。

### 解决方案

**方案 1：扩充对话引用检测词库**

```python
# app.py 中扩展 _DIALOGUE_REF
_DIALOGUE_REF = re.compile(
    r'(上一个|上一条|上一轮|上面|刚才|之前(说|问|讲|提到?)?|你说|你讲|你提到|'
    r'你回答|你刚|你之前|你(刚才|之前)(说|提)|我们(聊|说|讨论)|'
    r'对话(内容|记录)?|历史(记录)?|前面(说|提)|你(确定|确认)|'
    r'对吗|正确吗|有误|错了吗|再(说|讲|解释)一?下?|重复一下|'
    r'换[个种]方式|继续|你上面|你前面|解释一下刚|你说的|'
    r'那个(公式|方法|步骤|说法)|刚说的|更详细|举个例子|'       # ← 新增
    r'不对|重新(说|解释|回答)|你弄错了|错误|纠正)',             # ← 新增
    re.IGNORECASE
)
```

**方案 2：增加"智能路由"层——先判断是否需要检索**

这是更根本的解法：在检索之前，先用 LLM 判断这个问题是"知识库查询"还是"对话延续"：

```python
def _needs_kb_search(query: str, history: list) -> bool:
    """用一次轻量 LLM 调用判断是否需要检索知识库"""
    if not history:
        return True

    recent = history[-2:]   # 只看最近一轮
    history_text = "\n".join([
        f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:100]}"
        for m in recent
    ])

    prompt = f"""对话历史：
{history_text}

新问题：{query}

判断：这个问题是（A）需要查询专业知识库 还是（B）对上文对话的追问/延续？
只回复 A 或 B，不要解释。"""

    result = lightweight_llm.invoke(prompt)
    return result.content.strip().upper() == 'A'
```

**方案 3：对话历史压缩**

当历史超过一定长度时，用 LLM 把之前的对话压缩成摘要：

```python
def _compress_history(history: list, llm) -> list:
    """历史超过 2000 token 时，把前半段压缩成摘要"""
    if _count_tokens_total(history) < 2000:
        return history

    old = history[:-4]   # 保留最近 2 轮不压缩
    recent = history[-4:]

    summary_prompt = "用 3 句话总结以下对话的核心内容：\n" + \
                     "\n".join(m.content for m in old)
    summary = llm.invoke(summary_prompt).content

    return [SystemMessage(content=f"[之前对话摘要] {summary}")] + recent
```

---

## 痛点五：提示词工程——怎么写好、怎么管理

### 为什么 Prompt 这么难写？

一个最常见的认知误区：**让 LLM 帮你写 Prompt**。  
LLM 写出来的 Prompt 往往是"听起来专业但实际无效"的——因为它不知道你的具体问题在哪里。  
好的 Prompt 来自**观察模型的真实行为 → 找到偏差 → 针对性修正**，是迭代出来的，不是生成出来的。

---

### Prompt 的三层结构

参考 Coze/Dify 的设计思路，把 Prompt 拆成三层：

```
┌─────────────────────────────────────────┐
│  角色层（Role）                          │
│  你是谁、你的专业领域、你的行为原则       │
├─────────────────────────────────────────┤
│  规则层（Rules）                         │
│  使用资料的方式、禁止行为、输出格式要求   │
├─────────────────────────────────────────┤
│  变量层（Variables）                     │
│  {{domain}} {{audience}} {{detail_level}} │
│  运行时动态注入，无需改代码               │
└─────────────────────────────────────────┘
```

本项目已实现变量层（`prompt_templates/variables.json`），  
在 `system.txt` 中用 `{{变量名}}` 引用：

```
你是{{domain}}领域的{{role}}，面向{{audience}}。
```

---

### 高级 Prompt 管理技巧

**技巧 1：A/B 测试 Prompt**

本项目已有版本管理 API，利用它做对比测试：

```bash
# 保存当前版本 A
POST /api/prompts/save_version

# 修改 system.txt 为版本 B，热重载
POST /api/prompts/system
POST /api/prompts/reload

# 对同一组测试问题分别测试
# 效果好则保留 B，效果差则激活 A
POST /api/prompts/activate/system_版本A时间戳.txt
```

**技巧 2：Few-shot 示例注入**

在 system.txt 中放 2~3 个"好回答示例"，比任何文字描述都有效：

```
## 回答示例

问：镜头焦距怎么计算？
答：
焦距计算需要三个参数：
- f（焦距，mm）= WD × sensor_size / FOV
- WD：工作距离，指相机到被测物体的距离
- sensor_size：传感器尺寸（水平或垂直）
- FOV：视野范围

例：WD=500mm，sensor=6.4mm，FOV=100mm → f = 500×6.4/100 = 32mm
```

**技巧 3：思维链（Chain of Thought）激活**

在 Prompt 中要求模型"先分析，再作答"，可以显著减少胡说：

```
## 回答步骤
1. 先判断参考资料是否包含回答该问题所需的信息
2. 如有：梳理关键知识点，整合成有结构的回答
3. 如无：说明知识库中没有相关信息，给出通用建议
```

**技巧 4：用评分 Prompt 来评估 Prompt 质量**

```python
eval_prompt = """
以下是一个 RAG 系统的问答案例，请从 1-5 分评价回答质量：

问题：{question}
参考资料：{context}
实际回答：{answer}

评分标准：
5分 = 完全基于资料，有独立分析，格式清晰
3分 = 部分基于资料，但有不必要的填充内容
1分 = 忽视资料，或与资料矛盾

只输出分数（1-5）和一句改进建议。
"""
```

---

### 推荐的 Prompt 迭代流程

```
第一步：收集 10 个真实问题作为测试集
第二步：用当前 Prompt 跑一遍，记录回答
第三步：找出最差的 3 个答案，分析是哪类问题（照搬 / 忽视资料 / 套话）
第四步：针对这类问题，修改 Prompt 的对应规则（版本存档）
第五步：对比修改前后的分数，确认改善后提交
第六步：循环
```

不要追求一次写出完美 Prompt——每次迭代只解决一类问题，积累 5 轮后效果会质变。

---

## 总结：五大痛点的根本解法

| 痛点 | 根本原因 | 核心解法 |
|------|---------|---------|
| 召回失败 | 标题被切掉 / 阈值过高 / chunk 太碎 | 标题写入 chunk + 调低阈值 + 增大 chunk |
| 忽视资料 | context 注入位置错 / 指令太弱 | 参考资料放 system + 强化指令措辞 |
| 回答死板 | Prompt 用词暗示"复制" | 明确"原材料→成品"定位 + Few-shot 示例 |
| 上下文失忆 | 引用检测覆盖不全 / 缺检索路由 | 扩充引用词库 + 增加路由判断 |
| Prompt 难管理 | 缺乏迭代方法论 | 测试集 + A/B 存档 + CoT + 分层结构 |
