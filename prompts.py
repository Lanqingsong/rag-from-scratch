from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import json

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompt_templates")


def _apply_variables(text: str) -> str:
    """将 variables.json 中定义的变量插值到提示词文本中。"""
    vars_path = os.path.join(_PROMPT_DIR, "variables.json")
    if not os.path.exists(vars_path):
        return text
    try:
        with open(vars_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_vars = {**data.get("system_vars", {}), **data.get("user_vars", {})}
        for key, value in all_vars.items():
            text = text.replace(f"{{{{{key}}}}}", str(value))
    except Exception:
        pass
    return text


def _load(filename: str, fallback: str) -> str:
    path = os.path.join(_PROMPT_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return _apply_variables(text)
    return fallback


# ── System 提示词：简洁有效，只说"做什么"，不列"禁止清单" ──
_SYSTEM_DEFAULT = (
    "你是机器视觉领域的资深工程师AI助手。\n\n"
    "## 使用参考资料的方式\n"
    "系统消息中注入的'相关资料'是你推理的原材料，而不是答案模板。\n"
    "正确做法：理解资料含义 → 结合问题深入推理 → 用自己的语言系统作答，可以补充资料以外的专业知识。\n"
    "错误做法：原文复制资料内容、或以'根据资料/知识库'为开头引用资料。\n\n"
    "## 回答格式\n"
    "系统、深入、有结构——覆盖核心概念、计算公式、实际步骤和注意事项，使用标题和列表组织答案。\n"
    "若资料中缺乏某方面内容，主动补充你的专业知识，不要留白。\n\n"
    "## 对话历史\n"
    "若问题涉及当前对话本身，直接从对话历史中回答，不要检索知识库。"
)

_SYSTEM = _load("system.txt", _SYSTEM_DEFAULT)

# ── 模板：context 注入 system，human 只放问题（证明有效的结构） ──

KB_ONLY = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM + "\n\n## 相关资料\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{query}"),
])

DUAL_SOURCE = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM + "\n\n## 本地资料\n{kb_context}\n\n## 最新资讯\n{web_context}"),
    MessagesPlaceholder("history"),
    ("human", "{query}"),
])

WEB_ONLY = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM + "\n\n## 网络搜索结果\n{web_context}"),
    MessagesPlaceholder("history"),
    ("human", "{query}"),
])

GENERAL = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "{query}"),
])


def reload_prompts():
    global _SYSTEM, KB_ONLY, DUAL_SOURCE, WEB_ONLY, GENERAL
    _SYSTEM = _load("system.txt", _SYSTEM_DEFAULT)
    KB_ONLY = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM + "\n\n## 相关资料\n{context}"),
        MessagesPlaceholder("history"), ("human", "{query}"),
    ])
    DUAL_SOURCE = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM + "\n\n## 本地资料\n{kb_context}\n\n## 最新资讯\n{web_context}"),
        MessagesPlaceholder("history"), ("human", "{query}"),
    ])
    WEB_ONLY = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM + "\n\n## 网络搜索结果\n{web_context}"),
        MessagesPlaceholder("history"), ("human", "{query}"),
    ])
    GENERAL = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        MessagesPlaceholder("history"), ("human", "{query}"),
    ])
    print("提示词已热重载")
