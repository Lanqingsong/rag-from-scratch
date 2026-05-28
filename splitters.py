"""
文档分块策略模块

支持五种策略：
  markdown  - 按 Markdown 标题层级切割（适合有 ##/### 标题的文档）
  regex     - 用户自定义边界正则（适合编号式、Q&A 式文档）
  semantic  - Embedding 语义相似度切割（适合无结构文档）
  recursive - 标准递归字符切割（通用保底）
  auto      - 自动探测文档结构，按优先级选最合适的策略

每个知识库文件夹可放 kb_config.json 指定策略；无配置文件时使用 auto。
"""

import re
from typing import Optional
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────────────
# 公共清洗工具
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r'by\s+\d+@qq\.com\s*持续更新中…?', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'```mermaid[\s\S]*?```', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Markdown 标题切割
# ─────────────────────────────────────────────────────────────────────────────

class MarkdownHeadingSplitter:
    """
    按 Markdown 标题切割。
    heading_level=3 → 按 ### 切；2 → 按 ## 切；4 → 按 #### 切。
    超长 chunk 自动按下一级标题二次切割，最终回退到 RecursiveSplitter。
    """

    def __init__(self, heading_level: int = 3, chunk_size: int = 1000,
                 chunk_overlap: int = 0):
        self.level = heading_level
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 生成匹配目标层级及以上的正则
        prefix = '#' * heading_level
        self._pattern = re.compile(rf'^{prefix}\s+.+', re.MULTILINE)
        # 子标题用更深一级
        deeper = '#' * (heading_level + 1)
        self._sub_pattern = re.compile(rf'^{deeper}\s+.+', re.MULTILINE)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks = []
        for doc in documents:
            text = _clean(doc.page_content)
            chunks.extend(self._split(text, doc.metadata))
        return chunks

    def _split(self, text: str, metadata: dict) -> list[Document]:
        matches = list(self._pattern.finditer(text))
        if not matches:
            return _recursive_split(text, metadata, self.chunk_size, self.chunk_overlap)

        chunks = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            heading = m.group(0).strip().lstrip('#').strip()
            body = text[start:end].strip()
            # 去掉标题行本身前面的 # 号，保留标题文字在 chunk 里
            body = re.sub(r'^#{1,6}\s+', '', body, count=1)
            meta = {**metadata, 'section_title': heading}

            if len(body) > self.chunk_size:
                # 尝试按子标题二次切割
                sub = self._split_by_sub(body, meta)
                chunks.extend(sub)
            else:
                chunks.append(Document(page_content=body, metadata=meta))
        return chunks

    def _split_by_sub(self, text: str, metadata: dict) -> list[Document]:
        matches = list(self._sub_pattern.finditer(text))
        if not matches:
            return _recursive_split(text, metadata, self.chunk_size, self.chunk_overlap)
        chunks = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            body = re.sub(r'^#{1,6}\s+', '', body, count=1)
            heading = m.group(0).strip().lstrip('#').strip()
            meta = {**metadata, 'section_title': heading}
            if len(body) > self.chunk_size:
                chunks.extend(_recursive_split(body, meta, self.chunk_size, self.chunk_overlap))
            else:
                chunks.append(Document(page_content=body, metadata=meta))
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 2. 自定义正则边界切割
# ─────────────────────────────────────────────────────────────────────────────

class RegexBoundarySplitter:
    """
    用户提供一个正则，匹配每个 chunk 的"起始行"（如编号标题、Q&A 题目行）。
    示例 pattern：r'\\d+\\.\\d+\\s+[^\\n]+'  匹配 "1.1 xxx" 格式
                  r'Q\\d+[.:：]'              匹配 "Q1:" 格式
    """

    def __init__(self, pattern: str, chunk_size: int = 1000,
                 chunk_overlap: int = 0):
        self.pattern = re.compile(pattern, re.MULTILINE)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks = []
        for doc in documents:
            text = _clean(doc.page_content)
            chunks.extend(self._split(text, doc.metadata))
        return chunks

    def _split(self, text: str, metadata: dict) -> list[Document]:
        matches = list(self.pattern.finditer(text))
        if not matches:
            return _recursive_split(text, metadata, self.chunk_size, self.chunk_overlap)

        chunks = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            heading = m.group(0).strip()
            meta = {**metadata, 'section_title': heading}
            if len(body) > self.chunk_size:
                chunks.extend(_recursive_split(body, meta, self.chunk_size, self.chunk_overlap))
            else:
                chunks.append(Document(page_content=body, metadata=meta))
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 3. 语义切割（Embedding 相似度）
# ─────────────────────────────────────────────────────────────────────────────

class SemanticSplitter:
    """
    用相邻句子的 Embedding 余弦相似度找自然断点，相似度骤降处切割。
    threshold_type: 'percentile'（默认）| 'standard_deviation' | 'interquartile'
    threshold_value: percentile=95 意味着相似度下降最大的5%处切割
    """

    def __init__(self, embeddings, threshold_type: str = 'percentile',
                 threshold_value: float = 95, chunk_size: int = 1000):
        from langchain_experimental.text_splitter import SemanticChunker
        self._chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=threshold_type,
            breakpoint_threshold_amount=threshold_value,
        )
        self.chunk_size = chunk_size

    def split_documents(self, documents: list[Document]) -> list[Document]:
        cleaned = [Document(page_content=_clean(d.page_content),
                            metadata=d.metadata) for d in documents]
        try:
            return self._chunker.split_documents(cleaned)
        except Exception as e:
            print(f"[SemanticSplitter] 失败，回退到递归切割: {e}")
            chunks = []
            for doc in cleaned:
                chunks.extend(_recursive_split(doc.page_content, doc.metadata,
                                               self.chunk_size, 0))
            return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 4. 递归字符切割（通用保底）
# ─────────────────────────────────────────────────────────────────────────────

class RecursiveSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks = []
        for doc in documents:
            text = _clean(doc.page_content)
            chunks.extend(_recursive_split(text, doc.metadata,
                                           self.chunk_size, self.chunk_overlap))
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 5. Auto 策略：自动探测文档结构
# ─────────────────────────────────────────────────────────────────────────────

_MD_H3 = re.compile(r'^###\s+.+', re.MULTILINE)
_MD_H2 = re.compile(r'^##\s+.+', re.MULTILINE)
_MD_H4 = re.compile(r'^####\s+.+', re.MULTILINE)
_NUMBERED = re.compile(r'^\d+\.\d+\s+\S+', re.MULTILINE)
_NUMBERED_SIMPLE = re.compile(r'^\d+\.\s+\S+', re.MULTILINE)


def _detect_structure(text: str) -> tuple[str, dict]:
    """
    自动探测文档主要结构，返回 (strategy, kwargs)。
    探测优先级：H3 > H2 > H4 > 数字编号 > 递归
    """
    # 至少 3 个标题才认为是有效结构
    if len(_MD_H3.findall(text)) >= 3:
        return 'markdown', {'heading_level': 3}
    if len(_MD_H2.findall(text)) >= 3:
        return 'markdown', {'heading_level': 2}
    if len(_MD_H4.findall(text)) >= 3:
        return 'markdown', {'heading_level': 4}
    if len(_NUMBERED.findall(text)) >= 3:
        return 'regex', {'pattern': r'\d+\.\d+\s+[^\n]+'}
    if len(_NUMBERED_SIMPLE.findall(text)) >= 3:
        return 'regex', {'pattern': r'\d+\.\s+[^\n]+'}
    return 'recursive', {}


class AutoSplitter:
    """
    自动探测每份文档的结构，选择最合适的策略。
    有结构 → 结构切割；无结构 → 语义切割（需传入 embeddings）；
    语义切割不可用时回退到递归切割。
    """

    def __init__(self, embeddings=None, chunk_size: int = 1000,
                 chunk_overlap: int = 0):
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks = []
        for doc in documents:
            text = _clean(doc.page_content)
            strategy, kwargs = _detect_structure(text)
            splitter = _make(strategy, self.embeddings, self.chunk_size,
                             self.chunk_overlap, **kwargs)
            result = splitter.split_documents(
                [Document(page_content=text, metadata=doc.metadata)]
            )
            detected = f"{strategy}({kwargs})"
            src = doc.metadata.get('source', '')
            print(f"[AutoSplitter] {src} → {detected}，{len(result)} 块")
            chunks.extend(result)
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────────────────

def _recursive_split(text: str, metadata: dict, chunk_size: int,
                     chunk_overlap: int) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', '。', '！', '？', '；', ' ', ''],
    )
    return splitter.create_documents([text], metadatas=[metadata])


def _make(strategy: str, embeddings, chunk_size: int, chunk_overlap: int,
          **kwargs):
    """根据策略名创建对应切割器。"""
    if strategy == 'markdown':
        return MarkdownHeadingSplitter(
            heading_level=kwargs.get('heading_level', 3),
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
    if strategy == 'regex':
        return RegexBoundarySplitter(
            pattern=kwargs.get('pattern', r'\d+\.\d+\s+[^\n]+'),
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
    if strategy == 'semantic':
        if embeddings:
            return SemanticSplitter(
                embeddings=embeddings,
                threshold_type=kwargs.get('threshold_type', 'percentile'),
                threshold_value=kwargs.get('threshold_value', 95),
                chunk_size=chunk_size,
            )
        print("[Splitter] 语义切割需要 embeddings，回退到递归切割")
        return RecursiveSplitter(chunk_size=chunk_size,
                                 chunk_overlap=chunk_overlap)
    if strategy == 'recursive':
        return RecursiveSplitter(chunk_size=chunk_size,
                                 chunk_overlap=chunk_overlap)
    # auto / 默认
    return AutoSplitter(embeddings=embeddings, chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap)


def create_splitter(config: dict, embeddings=None):
    """
    从 kb_config.json 的配置 dict 创建切割器。

    示例配置：
      {"strategy": "markdown", "heading_level": 3, "chunk_size": 1000}
      {"strategy": "regex", "pattern": "\\d+\\.\\d+\\s+[^\\n]+"}
      {"strategy": "semantic", "threshold_type": "percentile", "threshold_value": 95}
      {"strategy": "auto"}   ← 默认
    """
    strategy = config.get('strategy', 'auto')
    chunk_size = int(config.get('chunk_size', 1000))
    chunk_overlap = int(config.get('chunk_overlap', 0))
    # 传递 config 中除公共字段外的剩余参数
    extra = {k: v for k, v in config.items()
             if k not in ('strategy', 'chunk_size', 'chunk_overlap',
                          'name', 'description')}
    return _make(strategy, embeddings, chunk_size, chunk_overlap, **extra)
