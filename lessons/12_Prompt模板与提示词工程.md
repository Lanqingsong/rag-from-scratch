# 12 · Prompt 模板与提示词工程

> 对应主项目 `prompts.py` 全文 + `prompt_templates/system.txt`

---

## Prompt 是 RAG 系统的"控制器"

检索出来的文档质量再好，如果 Prompt 写得差，LLM 的回答依然会令人失望。  
Prompt 决定了：
- 模型扮演什么角色（角色设定）
- 如何使用检索到的文档（使用规则）
- 回答的格式和风格（输出约束）

---

## ChatPromptTemplate 结构

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

KB_ONLY = ChatPromptTemplate.from_messages([
    ("system", system_text + "\n\n## 相关资料\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{query}"),
])
```

**三段式结构的设计原因：**

| 位置 | 角色 | 内容 |
|------|------|------|
| system | 系统指令 | 角色设定 + 参考资料注入（放 system 而非 human，模型遵循度更高） |
| history | 对话历史 | 自动展开为多条 human/ai 交替消息 |
| human | 用户问题 | 只放问题本身，简洁清晰 |

**为什么把文档注入 system 而不是 human？**  
实验表明，LLM 对 system 消息的遵循程度高于 human 消息，  
把"参考资料"放在 system 能减少模型忽略文档的概率。

---

## 4 种 Prompt 模式

```python
# 1. 只有本地知识库
KB_ONLY = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + "\n\n## 相关资料\n{context}"),
    ...
])

# 2. 本地 + 网络双源
DUAL_SOURCE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + "\n\n## 本地资料\n{kb_context}\n\n## 最新资讯\n{web_context}"),
    ...
])

# 3. 只有网络搜索
WEB_ONLY = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + "\n\n## 网络搜索结果\n{web_context}"),
    ...
])

# 4. 无上下文（直接问 LLM）
GENERAL = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ...
])
```

---

## System Prompt 的设计原则

本项目的 `system.txt` 内容：

```
你是机器视觉领域的资深工程师AI助手。

## 使用参考资料的方式
系统消息中注入的"相关资料"是你推理的原材料，而不是答案模板。
正确做法：理解资料含义 → 结合问题深入推理 → 用自己的语言系统作答，可以补充资料以外的专业知识。
错误做法：原文复制资料内容、或以"根据资料/知识库"为开头引用资料。

## 回答格式
系统、深入、有结构——覆盖核心概念、计算公式、实际步骤和注意事项，使用标题和列表组织答案。
```

**关键设计决策：**

1. **说"要做什么"，不列"禁止清单"**  
   "用自己的语言系统作答"比"不要照搬原文"效果更好——正向指令比负向约束更有效。

2. **给出正确/错误示范**  
   对比示例比单纯说明更容易被 LLM 理解和遵循。

3. **格式约束放在独立章节**  
   用 `## 标题` 分隔不同类型的指令，有助于 LLM 清晰理解各指令的作用域。

---

## 变量插值系统

`variables.json` 中定义的变量可以在 `system.txt` 中用 `{{变量名}}` 引用，  
无需修改代码就能动态调整提示词：

```json
// prompt_templates/variables.json
{
  "system_vars": {
    "domain": "机器视觉",
    "role":   "资深工程师AI助手"
  },
  "user_vars": {
    "detail_level": "expert"
  }
}
```

```
// prompt_templates/system.txt
你是{{domain}}领域的{{role}}。
```

**实现原理：**

```python
def _apply_variables(text: str) -> str:
    vars_path = os.path.join(_PROMPT_DIR, "variables.json")
    with open(vars_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_vars = {**data.get("system_vars", {}), **data.get("user_vars", {})}
    for key, value in all_vars.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    return text
```

注意双重花括号：`{{key}}` 在 Python f-string 中表示字面量 `{key}`，  
而 LangChain 模板使用单花括号 `{key}` 作为变量占位符，两者不冲突。

---

## 热重载机制

修改 `system.txt` 后，不需要重启服务，调用 API 即可立即生效：

```python
def reload_prompts():
    global _SYSTEM, KB_ONLY, DUAL_SOURCE, WEB_ONLY, GENERAL
    _SYSTEM = _load("system.txt", _SYSTEM_DEFAULT)   # 重新读文件
    # 重新构建 4 个模板（引用新的 _SYSTEM 变量）
    KB_ONLY = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM + "\n\n## 相关资料\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{query}"),
    ])
    # ... 其他 3 个同理

# app.py 中调用
@app.route('/api/prompts/reload', methods=['POST'])
def reload_prompts_api():
    _p.reload_prompts()
    llm = LLMClient()   # 用新提示词重建 LCEL 链
    return jsonify({"status": "ok"})
```

---

## 版本管理 API

```
GET  /api/prompts/versions              → 列出所有历史版本文件名
POST /api/prompts/save_version          → 将当前 system.txt 存档（加时间戳）
POST /api/prompts/activate/<version>    → 激活某历史版本并热重载
POST /api/prompts/system                → 直接覆盖 system.txt 内容并热重载
```

这套机制让调 Prompt 变得安全：先存档再修改，效果不好随时回滚。

---

## 动手练习

1. 修改 `prompt_templates/system.txt`，改变模型的回答风格（如更简洁、或更学术）
2. 调用 `POST /api/prompts/reload` 热重载，对比前后回答差异
3. 给 `variables.json` 添加一个 `audience` 变量，在 system.txt 中用 `{{audience}}` 引用，  
   动态切换"面向初学者"和"面向专家"的回答深度
4. 思考：把参考资料放在 system 里，和放在 human 消息里，有什么区别？
