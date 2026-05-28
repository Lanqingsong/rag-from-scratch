import os
import json
import shutil
import re
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_core.documents import Document
from config import Config
from splitters import create_splitter

try:
    import dashscope
    from dashscope import TextEmbedding
    HAS_DASHSCOPE = True
except ImportError:
    HAS_DASHSCOPE = False
    print("Warning: dashscope not installed, will use OpenAI embeddings fallback")
    from langchain_openai import OpenAIEmbeddings

try:
    from langchain_community.retrievers import BM25Retriever
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("Warning: rank_bm25 未安装，BM25 混合检索已禁用。运行: pip install rank_bm25")

from langchain_core.embeddings import Embeddings


class QianwenEmbeddings(Embeddings):
    def __init__(self):
        if not HAS_DASHSCOPE:
            raise ImportError("dashscope is required for Qianwen embeddings")
        dashscope.api_key = Config.QIANWEN_API_KEY
        self.batch_size = 25

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            texts = [texts]
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = TextEmbedding.call(
                model=Config.EMBEDDING_MODEL,
                input=batch
            )
            if response.status_code == 200:
                batch_embeddings = [r['embedding'] for r in response.output['embeddings']]
                all_embeddings.extend(batch_embeddings)
            else:
                raise Exception(f"Embedding API error: {response.message}")
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

class QianwenReranker:
    def __init__(self):
        if not HAS_DASHSCOPE:
            raise ImportError("dashscope is required for Qianwen rerank")
        dashscope.api_key = Config.QIANWEN_API_KEY
    
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
        else:
            raise Exception(f"Rerank API error: {response.message}")

class QATextSplitter:
    def __init__(self, chunk_size=1200, chunk_overlap=0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _clean_text(self, text):
        text = re.sub(r'by\s+\d+@qq\.com\s*持续更新中…?', '', text)
        text = re.sub(r'by\s+\d+@qq\.com', '', text)
        text = re.sub(r'持续更新中…?', '', text)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\$\$', '', text)
        text = re.sub(r'\\text\{(.*?)\}', r'\1', text)
        text = re.sub(r'```mermaid[\s\S]*?```', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def split_documents(self, documents):
        all_chunks = []
        
        for doc in documents:
            text = self._clean_text(doc.page_content)
            if not text:
                continue
            is_md = doc.metadata.get('source', '').endswith('.md')
            if is_md:
                chunks = self._split_by_markdown_headings(text, doc.metadata)
            else:
                chunks = self._split_by_questions(text, doc.metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _split_by_markdown_headings(self, text, metadata):
        chunks = []
        pattern = r'(###\s+\d+\.\s+[^\n]+)'
        matches = list(re.finditer(pattern, text))

        if not matches:
            pattern = r'(####\s+\d+\.\d+\s+[^\n]+)'
            matches = list(re.finditer(pattern, text))

        if not matches:
            chunks.append(Document(page_content=text, metadata=metadata))
            return chunks

        for i, match in enumerate(matches):
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_title = re.sub(r'^#{2,4}\s+', '', match.group(0)).strip()
            chunk_text = text[start_pos:end_pos].strip()
            chunk_text = re.sub(r'^#{2,4}\s+', '', chunk_text)
            chunk_meta = {**metadata, 'section_title': section_title}

            if len(chunk_text) > self.chunk_size:
                sub_chunks = self._split_by_sub_headings(chunk_text, chunk_meta)
                chunks.extend(sub_chunks)
            else:
                chunks.append(Document(page_content=chunk_text, metadata=chunk_meta))

        return chunks
    
    def _split_by_sub_headings(self, text, metadata):
        chunks = []
        pattern = r'(####\s+\d+\.\d+\s+[^\n]+)'
        matches = list(re.finditer(pattern, text))
        
        if not matches:
            return self._split_long_chunk(text, metadata)
        
        for i, match in enumerate(matches):
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_text = text[start_pos:end_pos].strip()
            chunk_text = re.sub(r'^#{2,4}\s+', '', chunk_text)
            
            if len(chunk_text) > self.chunk_size:
                sub_chunks = self._split_long_chunk(chunk_text, metadata)
                chunks.extend(sub_chunks)
            else:
                chunks.append(Document(page_content=chunk_text, metadata=metadata))
        
        return chunks
    
    def _split_by_questions(self, text, metadata):
        chunks = []
        
        pattern = r'(\d+\.\d+\s+[^\n]+)'
        matches = list(re.finditer(pattern, text))
        
        if not matches:
            pattern_fallback = r'(\d+\.\s+[^\n]+)'
            matches = list(re.finditer(pattern_fallback, text))
        
        if not matches:
            chunks.append(Document(page_content=text, metadata=metadata))
            return chunks
        
        for i, match in enumerate(matches):
            start_pos = match.start()
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            chunk_text = text[start_pos:end_pos].strip()
            
            if len(chunk_text) > self.chunk_size:
                sub_chunks = self._split_long_chunk(chunk_text, metadata)
                chunks.extend(sub_chunks)
            else:
                chunks.append(Document(page_content=chunk_text, metadata=metadata))
        
        return chunks
    
    def _split_long_chunk(self, text, metadata):
        chunks = []
        
        sentences = re.split(r'([。！？\n])', text)
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
            else:
                sentence = sentences[i]
            
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(Document(page_content=current_chunk.strip(), metadata=metadata))
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(Document(page_content=current_chunk.strip(), metadata=metadata))
        
        return chunks

class KnowledgeBase:
    def __init__(self):
        self.use_qianwen = HAS_DASHSCOPE and Config.QIANWEN_API_KEY

        if self.use_qianwen:
            self.embeddings = QianwenEmbeddings()
            self.reranker = QianwenReranker()
        else:
            # dashscope 未安装时降级到 OpenAI Embeddings 接口
            # 注意：DeepSeek 不提供 Embedding 服务，此降级仅适用于持有 OpenAI API Key 的用户。
            # 若只有 DeepSeek Key，请安装 dashscope 并配置 QIANWEN_API_KEY。
            print("Warning: dashscope 未安装，Embedding 降级到 OpenAI 兼容接口（需真实 OpenAI Key）")
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=Config.DEEPSEEK_API_KEY,
                base_url=Config.DEEPSEEK_API_BASE
            )
            self.reranker = None

        self.vector_store = None
        self.bm25_retriever = None
        
    def load_documents(self):
        if not os.path.exists(Config.KNOWLEDGE_BASE_DIR):
            os.makedirs(Config.KNOWLEDGE_BASE_DIR)

        all_chunks = []

        # 根目录文档 + 所有子目录，分别按各自的 kb_config.json 切割
        scan_dirs = [Config.KNOWLEDGE_BASE_DIR]
        for entry in os.scandir(Config.KNOWLEDGE_BASE_DIR):
            if entry.is_dir():
                scan_dirs.append(entry.path)

        found_any = False
        for directory in scan_dirs:
            docs = self._load_raw_documents(directory)
            if not docs:
                continue
            found_any = True
            cfg = self._load_kb_config(directory)
            splitter = create_splitter(cfg, self.embeddings)
            name = os.path.basename(directory) or 'root'
            chunks = splitter.split_documents(docs)
            print(f"[KB] {name}: {len(docs)} 文件 → {len(chunks)} 块"
                  f"（策略: {cfg.get('strategy','auto')}）")
            all_chunks.extend(chunks)

        if not found_any:
            self._create_sample_document()
            docs = self._load_raw_documents(Config.KNOWLEDGE_BASE_DIR)
            cfg = self._load_kb_config(Config.KNOWLEDGE_BASE_DIR)
            all_chunks = create_splitter(cfg, self.embeddings).split_documents(docs)

        return all_chunks

    def _load_raw_documents(self, directory: str) -> list:
        """加载指定目录下的文档（不递归进子目录）。"""
        docs = []
        for ext, loader_cls, kwargs in [
            ('*.txt', TextLoader, {'encoding': 'utf-8'}),
            ('*.md',  TextLoader, {'encoding': 'utf-8'}),
        ]:
            try:
                loader = DirectoryLoader(
                    directory, glob=ext, loader_cls=loader_cls,
                    loader_kwargs=kwargs, recursive=False,
                )
                docs.extend(loader.load())
            except Exception as e:
                print(f"Warning: 加载 {directory}/{ext} 失败: {e}")
        try:
            loader = DirectoryLoader(
                directory, glob='*.pdf', loader_cls=PyPDFLoader, recursive=False,
            )
            docs.extend(loader.load())
        except Exception as e:
            print(f"Warning: 加载 {directory}/*.pdf 失败: {e}")
        return docs

    @staticmethod
    def _load_kb_config(directory: str) -> dict:
        """读取目录下的 kb_config.json，不存在时返回 auto 默认配置。"""
        path = os.path.join(directory, 'kb_config.json')
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: 读取 {path} 失败: {e}")
        return {'strategy': 'auto', 'chunk_size': 1000, 'chunk_overlap': 0}
    
    def _create_sample_document(self):
        sample_content = """欢迎来到我的知识库

公司介绍
--------
本公司成立于2024年，专注于人工智能和大数据领域的研发与应用。

产品服务
--------
我们提供以下核心服务：
1. 智能客服系统 - 基于大语言模型的智能问答服务
2. 数据分析平台 - 实时数据处理和可视化分析
3. 机器学习解决方案 - 定制化的AI模型开发

技术优势
--------
- 拥有自主研发的大模型微调技术
- 支持多模态数据处理
- 提供私有化部署方案

联系方式
--------
邮箱: contact@example.com
电话: 400-123-4567
地址: 北京市朝阳区科技园区

常见问题
--------
Q: 如何开始使用服务？
A: 请访问我们的官网注册账号，或联系销售团队获取帮助。

Q: 是否支持私有化部署？
A: 是的，我们提供完整的私有化部署方案，确保数据安全。

Q: 服务费用如何计算？
A: 我们提供按需付费和包年订阅两种模式，具体价格请咨询销售。
"""
        sample_path = os.path.join(Config.KNOWLEDGE_BASE_DIR, "sample.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(sample_content)
    
    def add_pdf_to_knowledge_base(self, pdf_path):
        if not os.path.exists(pdf_path):
            print(f"PDF文件不存在: {pdf_path}")
            return False
        
        if not os.path.exists(Config.KNOWLEDGE_BASE_DIR):
            os.makedirs(Config.KNOWLEDGE_BASE_DIR)
        
        filename = os.path.basename(pdf_path)
        dest_path = os.path.join(Config.KNOWLEDGE_BASE_DIR, filename)
        
        shutil.copy2(pdf_path, dest_path)
        print(f"已将PDF文件添加到知识库: {filename}")
        return True
    
    def build_vector_store(self):
        docs = self.load_documents()
        if docs:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self._save_vector_store()
            self._build_bm25(docs)
            print(f"已构建向量数据库，共 {len(docs)} 个文档片段")
        else:
            print("未找到知识库文档，使用空向量库")

    def _save_vector_store(self):
        if not os.path.exists(Config.VECTOR_STORE_DIR):
            os.makedirs(Config.VECTOR_STORE_DIR)
        self.vector_store.save_local(Config.VECTOR_STORE_DIR)

    def load_vector_store(self):
        if os.path.exists(Config.VECTOR_STORE_DIR):
            try:
                self.vector_store = FAISS.load_local(
                    Config.VECTOR_STORE_DIR,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("已加载本地向量数据库")
                self._build_bm25_from_store()
                return True
            except Exception as e:
                print(f"加载向量数据库失败: {e}")
                return False
        return False

    def _build_bm25(self, docs):
        """从文档列表构建 BM25 索引。"""
        if not HAS_BM25 or not docs:
            return
        try:
            self.bm25_retriever = BM25Retriever.from_documents(
                docs, k=Config.TOP_K_RESULTS * 2
            )
            print(f"BM25 索引已建立，共 {len(docs)} 个文档")
        except Exception as e:
            print(f"BM25 构建失败: {e}")

    def _build_bm25_from_store(self):
        """从已加载的 FAISS docstore 提取文档并构建 BM25。"""
        if not HAS_BM25 or not self.vector_store:
            return
        try:
            all_docs = list(self.vector_store.docstore._dict.values())
            self._build_bm25(all_docs)
        except Exception as e:
            print(f"BM25 从向量库恢复失败: {e}")
    
    def search(self, query, k=Config.TOP_K_RESULTS, llm=None):
        if not self.vector_store:
            return []

        # ── FAISS 语义检索（带相关度阈值过滤）──────────────────
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": Config.KB_RELEVANCE_SCORE, "k": k}
        )
        if Config.MULTI_QUERY_RETRIEVAL and llm:
            retriever = self._build_multi_query_retriever(base_retriever, llm)
        else:
            retriever = base_retriever

        faiss_results = retriever.invoke(query)

        # ── BM25 关键词检索（精确词汇/标题命中）────────────────
        bm25_results = []
        if self.bm25_retriever:
            try:
                bm25_results = self.bm25_retriever.invoke(query)
            except Exception as e:
                print(f"BM25 检索失败: {e}")

        print(f"KB 命中：FAISS={len(faiss_results)} BM25={len(bm25_results)}")

        # ── RRF 融合两路结果 ──────────────────────────────────
        results = self._rrf_merge(faiss_results, bm25_results, k * 2)

        # ── 去重 ──────────────────────────────────────────────
        seen, unique = set(), []
        for doc in results:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        results = unique

        # ── Rerank 精排 ───────────────────────────────────────
        if self.reranker and len(results) > 1:
            try:
                results = self.reranker.rerank_documents(query, results)
                print(f"Rerank 完成，保留前 {len(results)} 条")
            except Exception as e:
                print(f"Rerank 失败，使用原始结果: {e}")

        return results

    def _rrf_merge(self, faiss_docs, bm25_docs, k, rrf_k=60):
        """Reciprocal Rank Fusion：合并两路排名列表。"""
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

    def _build_multi_query_retriever(self, base_retriever, llm):
        try:
            from langchain.retrievers import MultiQueryRetriever
        except ImportError:
            from langchain_community.retrievers import MultiQueryRetriever
        from langchain_core.prompts import PromptTemplate

        prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "你是一位信息检索专家。请为以下问题生成3个不同角度的查询语句，"
                "帮助从知识库中检索到更全面的信息。\n"
                "原始问题：{question}\n"
                "直接输出3个查询语句，每行一个，不要编号或其他格式："
            )
        )
        try:
            return MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm,
                prompt=prompt,
                include_original=True
            )
        except Exception as e:
            print(f"MultiQueryRetriever 构建失败，回退到单次检索: {e}")
            return base_retriever