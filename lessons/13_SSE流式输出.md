# 13 · SSE 流式输出

> 对应主项目 `app.py` 中 `chat_stream()` 路由（约 486~563 行）

---

## 为什么需要流式输出？

大模型生成一段 300 字的回答可能需要 5~10 秒。  
如果等全部生成完再返回，用户面对的是 5~10 秒的**白屏等待**。

**流式输出（Streaming）：** 每生成一个 token（通常 1~3 个字）就立刻推送给前端，  
用户看到文字像打字机一样逐渐出现，体验完全不同。

---

## SSE 协议

**Server-Sent Events** 是 HTTP 标准的一部分，让服务器向客户端**单向推送**事件流。  
格式极其简单：

```
data: {"type": "token", "content": "你"}\n\n
data: {"type": "token", "content": "好"}\n\n
data: {"type": "done", "answer_html": "<p>...</p>"}\n\n
data: [DONE]\n\n
```

- 每条消息以 `data: ` 开头
- 以两个换行 `\n\n` 结束
- 前端用 `EventSource` API 接收，无需 WebSocket

---

## Flask 中实现 SSE

```python
from flask import Response

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():

    def generate():           # 生成器函数，yield 每一条 SSE 事件
        yield _sse({'type': 'status', 'content': '正在检索...'})
        # ... 检索 ...
        yield _sse({'type': 'refs', 'references': [...]})
        for chunk in llm.stream(query, context, web_context, history):
            yield _sse({'type': 'token', 'content': chunk})
        yield _sse({'type': 'done', 'answer_html': rendered_html})
        yield "data: [DONE]\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',    # 禁用 Nginx 缓冲，否则会攒批再推
        }
    )

def _sse(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
```

---

## 事件类型设计

本项目定义了 5 种事件，每种有明确的时机和语义：

| 事件 | 触发时机 | 前端行为 |
|------|---------|---------|
| `status` | 检索开始前 | 显示"正在检索..."状态提示 |
| `refs` | 检索完成后、LLM 开始前 | 立刻展示参考资料，用户无需等待回答 |
| `token` | LLM 每生成一段文本 | 追加到回答区域，实时显示 |
| `done` | LLM 生成完毕 | 用服务端渲染的 HTML 替换原始文本（公式/代码正确渲染） |
| `error` | 任何步骤出错 | 显示错误信息 |

**为什么先推 `refs` 再开始生成？**  
检索是毫秒级完成的，而 LLM 可能需要几秒才开始输出第一个 token。  
把参考资料先推出去，用户在等待时就能看到"系统找到了什么"，降低焦虑。

---

## 回答开头套话的自动清理

LLM 常常以"根据知识库内容，..."开头，这类套话对用户没有价值。  
本项目用缓冲区延迟推送，等积累到 40 个字符后再过滤：

```python
buf = ""
buf_flushed = False
BUF_SIZE = 40   # 足够覆盖最长的"根据..."开头

for chunk in llm.stream(...):
    if not buf_flushed:
        buf += chunk
        if len(buf) >= BUF_SIZE:
            buf = _clean(buf)        # 过滤套话
            buf_flushed = True
            yield _sse({'type': 'token', 'content': buf})
            buf = ""
    else:
        yield _sse({'type': 'token', 'content': chunk})

# 处理极短回答（buf 未被 flush 的情况）
if buf:
    yield _sse({'type': 'token', 'content': _clean(buf)})
```

**套话正则：**

```python
_BAD_OPENER = re.compile(
    r'^[\s\n]*(根据(提供的|以上|本|这些)?(知识库|资料|搜索结果|参考)(内容|信息)?'
    r'[，,。]?\s*|知识库(中)?(提到|说明|显示)[，,]?\s*)',
    re.IGNORECASE
)
```

---

## 对话引用检测：跳过检索

用户有时问的不是知识库中的内容，而是关于当前对话本身：

```
"你上一条说的公式是什么？"
"刚才的回答能换个角度再解释一次吗？"
```

检索知识库对这类问题没有意义，还浪费时间。  
本项目通过正则识别这类引用，直接走对话历史：

```python
_DIALOGUE_REF = re.compile(
    r'(上一个|上一条|刚才|之前(说|问|讲)?|你说|你回答|你之前|'
    r'对话(内容|记录)?|前面(说|提)|继续|解释一下刚)',
    re.IGNORECASE
)
_SHORT_PRONOUN = re.compile(r'(这个|那个|它|这些|那些)')

def _is_dialogue_ref(query, history):
    if not history:         # 没有历史记录，不可能是对话引用
        return False
    if _DIALOGUE_REF.search(query):
        return True
    if len(query) <= 15 and _SHORT_PRONOUN.search(query):
        return True         # 短问题 + 指代词，大概率是历史引用
    return False
```

---

## Markdown + LaTeX 渲染

LLM 生成的文本可能包含 Markdown 格式和数学公式，  
直接显示源码对用户不友好，需要服务端渲染成 HTML：

```python
def render_markdown(text):
    # 步骤 1：保存代码块（防止被 markdown 库二次处理）
    code_blocks = []
    text = re.sub(r'```[\s\S]*?```', lambda m: (code_blocks.append(m.group(0)),
                  f'\x00CODE{len(code_blocks)-1}\x00')[1], text)

    # 步骤 2：保存数学公式（保护 $ 符号不被转义）
    block_math, inline_math = [], []
    text = re.sub(r'\$\$(.+?)\$\$', save_block, text, flags=re.DOTALL)
    text = re.sub(r'\$(.+?)\$',     save_inline, text)

    # 步骤 3：Markdown → HTML
    html = markdown.markdown(text, extensions=['extra', 'fenced_code', 'codehilite'])

    # 步骤 4：还原公式为 MathJax 标签
    for i, f in enumerate(inline_math):
        html = html.replace(f'\x00IMATH{i}\x00',
            f'<span class="math-inline">\\({f}\\)</span>')

    # 步骤 5：还原代码块
    ...
    return html
```

**为什么要先"保存"再"还原"？**  
Markdown 库会把 `$` 转义成 HTML 实体，导致 MathJax 无法识别公式。  
先用占位符替换，处理完再替换回来，是处理多种标记语言混合时的标准技巧。

---

## 动手练习

1. 用浏览器开发者工具的 Network 面板，观察 `/api/chat/stream` 的 EventStream 响应
2. 修改 `BUF_SIZE`，观察不同缓冲大小对套话过滤的影响
3. 在 `_DIALOGUE_REF` 正则中添加新的触发词，测试效果
4. 思考：如果 LLM API 本身不支持流式，如何模拟"流式"体验？
