# 11 · LCEL 链式编程

> 对应主项目 `llm_client.py` 全文 + `prompts.py`

---

## 什么是 LCEL？

**LCEL（LangChain Expression Language）** 是 LangChain 0.1 引入的声明式管道语法，  
用 `|`（管道符）把多个组件串联成一条链，数据从左向右流动。

```python
chain = prompt | llm | parser

# 等价于：
result = parser.invoke(llm.invoke(prompt.invoke(inputs)))
```

这种写法的好处：每个组件职责单一，可以独立替换，天然支持流式传输。

---

## 三个核心组件

### 1. Prompt 模板 — 构建输入

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "你是一名机器视觉工程师AI助手。\n\n## 参考资料\n{context}"),
    MessagesPlaceholder("history"),   # 对话历史占位符
    ("human", "{query}"),
])
```

`MessagesPlaceholder("history")` 会把传入的 `history` 列表（`HumanMessage` / `AIMessage` 对象列表）  
按顺序展开插入，实现多轮对话上下文传递。

### 2. LLM — 生成回答

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=Config.MODEL_NAME,
    api_key=Config.DEEPSEEK_API_KEY,
    base_url=Config.DEEPSEEK_API_BASE,
    temperature=Config.TEMPERATURE,
    max_tokens=Config.MAX_TOKENS,
)
```

### 3. 输出解析器 — 提取纯文本

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
# 作用：从 LLM 返回的 AIMessage 对象中提取 .content 字符串
```

---

## 组合成链

```python
chain = template | llm | parser
```

**数据流：**

```
inputs (dict)
    ↓ template.invoke(inputs) → 格式化后的 ChatPromptValue（消息列表）
    ↓ llm.invoke(messages)    → AIMessage（含 .content 和 token 用量）
    ↓ parser.invoke(msg)      → str（纯文本）
result
```

---

## 本项目的 4 种链

```python
# llm_client.py
self.kb_chain      = KB_ONLY      | self.llm | parser  # 只有本地知识库上下文
self.dual_chain    = DUAL_SOURCE  | self.llm | parser  # 本地 + 网络双源
self.web_chain     = WEB_ONLY     | self.llm | parser  # 只有网络搜索结果
self.general_chain = GENERAL      | self.llm | parser  # 无上下文，直接问 LLM
```

**为什么要 4 种而不是 1 种？**

每种 Prompt 模板的变量不同：
- `KB_ONLY` 期望 `{context}` 变量
- `DUAL_SOURCE` 期望 `{kb_context}` 和 `{web_context}`
- `GENERAL` 没有上下文变量

如果用一个模板，当某个变量不存在时 LangChain 会报错。  
4 种链 + `_pick()` 选择逻辑是最干净的解法：

```python
def _pick(self, kb, web):
    if kb and web:  return self.dual_chain,    {"kb_context": ..., "web_context": ...}
    if kb:          return self.kb_chain,      {"context": ...}
    if web:         return self.web_chain,     {"web_context": ...}
    return          self.general_chain,        {}
```

---

## invoke vs stream

```python
# invoke：等全部 token 生成完毕后一次性返回
answer = chain.invoke({"query": "...", "context": "...", "history": []})
# answer 是 str

# stream：每生成一个 token 就 yield 出来
for chunk in chain.stream({"query": "...", "context": "...", "history": []}):
    print(chunk, end="", flush=True)
# chunk 是 str（每次可能是 1 个或几个 token）
```

**本项目 Web 接口使用 `stream`，CLI 使用 `invoke`。**

---

## 模型热切换

当用户在 Web 界面更换模型后，只需重建 `LLMClient` 实例即可：

```python
# app.py
@app.route('/api/config/llm', methods=['POST'])
def set_llm_config():
    global llm
    Config.save_runtime(data)    # 写入 model_config.json
    llm = LLMClient()            # 用新配置重新构建 4 条链
    return jsonify({"status": "ok"})
```

LCEL 链的重建代价极小（不需要重新加载模型权重，只是重新初始化 Python 对象），  
所以热切换可以做到无感知。

---

## 动手练习

1. 打开 `llm_client.py`，单独实例化一条链并调用 `invoke`
2. 修改 `_pick()` 逻辑，添加第 5 种情况：KB 命中但 web 搜索结果极少时仍走 `kb_chain`
3. 用 `stream` 方法实现一个简单的打字机效果命令行聊天
