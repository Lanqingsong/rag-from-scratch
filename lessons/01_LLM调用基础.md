# 01 · LLM 调用基础

> 对应代码：`01call_llm.py` / `models.py`

---

## 大语言模型的接口是什么？

大模型对外提供的是一个 **HTTP 接口**，发送一段对话消息，返回模型生成的文本。  
OpenAI 率先制定了这套接口规范，现在 DeepSeek、千问、Ollama 等几乎所有模型都兼容这个格式，  
所以同一套代码可以切换不同的模型提供商。

```
POST https://api.deepseek.com/v1/chat/completions

{
  "model": "deepseek-chat",
  "messages": [
    {"role": "system", "content": "你是一个翻译助手"},
    {"role": "user",   "content": "你好，今天过得好吗？"}
  ]
}
```

返回：

```json
{
  "choices": [{
    "message": {"role": "assistant", "content": "Hello, how are you today?"}
  }]
}
```

---

## 为什么用 LangChain 而不直接调用 HTTP？

直接调用 HTTP 没有问题，但当你需要把 LLM 和其他组件（检索器、解析器、提示词模板）组合成流水线时，  
手写胶水代码会越来越复杂。LangChain 提供了一套统一的抽象，让这些组件可以像管道一样拼接。

本项目中 LangChain 的作用：

```
知识库检索结果
     ↓
 Prompt 模板（注入上下文）
     ↓
 ChatOpenAI（调用 LLM）
     ↓
 StrOutputParser（提取纯文本）
     ↓
     回答
```

---

## 核心代码解读

### `models.py` — 统一封装客户端

```python
from langchain_openai import ChatOpenAI
import os

def get_lc_model_client(model="qwen-max-latest"):
    return ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=model,
    )
```

`ChatOpenAI` 是 LangChain 对 OpenAI 兼容接口的封装。  
`base_url` 参数让你把请求指向任意兼容 OpenAI 协议的服务。

---

### `01call_llm.py` — 最小调用示例

```python
client = get_lc_model_client()

msg = [
    ("system", "请将以下的内容翻译成英文"),
    ("human",  "你好，你今天过得好吗？"),
]

result = client.invoke(msg)
print(result.content)   # "Hello, how are you today?"
```

**消息格式说明：**

| 角色 | 含义 |
|------|------|
| `system` | 设定模型行为的系统指令（不参与对话显示） |
| `human` | 用户的输入 |
| `ai` | 模型之前的回复（多轮对话时使用） |

---

## `invoke` vs `stream` vs `batch`

```python
# invoke：等待完整回答后返回
result = client.invoke(msg)

# stream：逐 token 返回，适合流式展示
for chunk in client.stream(msg):
    print(chunk.content, end="", flush=True)

# batch：并行处理多个请求
results = client.batch([msg1, msg2, msg3])
```

本项目的 Web 界面使用 `stream`，避免用户等待时间过长。

---

## 动手练习

1. 运行 `01call_llm.py`，观察输出结果
2. 修改 `system` 消息，改为"你是一个诗人，请将内容改写为七言绝句"
3. 将 `invoke` 改为 `stream`，观察逐字输出的效果
4. 换一个模型（如 `deepseek-chat`），对比回答差异

---

## 主项目中的对应代码

`llm_client.py` — LLMClient 类，组装了 4 种不同的 LCEL 链（KB专用、双源、仅网络、通用），  
根据是否有知识库上下文自动选择合适的链路。
