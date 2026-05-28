import os
import json
from dotenv import load_dotenv

load_dotenv()

_RUNTIME_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "model_config.json")


class Config:
    # ── 默认值从 .env 读取 ──────────────────────────────────
    DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

    QIANWEN_API_KEY  = os.getenv("QIANWEN_API_KEY")
    QIANWEN_API_BASE = os.getenv("QIANWEN_API_BASE", "https://dashscope.aliyuncs.com/api/v1")

    KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
    VECTOR_STORE_DIR   = os.path.join(os.path.dirname(__file__), "vector_store")

    MODEL_NAME      = "deepseek-v4-pro"
    EMBEDDING_MODEL = "text-embedding-v2"
    TEMPERATURE     = 0.7
    MAX_TOKENS      = 2048

    MAX_CONTEXT_TOKENS  = 4096
    TOP_K_RESULTS       = 5
    RERANK_TOP_K        = 3

    WEB_SEARCH_ENABLED     = True
    WEB_SEARCH_MAX_RESULTS = 3
    SEARXNG_URL            = os.getenv("SEARXNG_URL", "http://localhost:8080")
    SEARXNG_LANGUAGE       = os.getenv("SEARXNG_LANGUAGE", "zh-CN")

    MAX_HISTORY_TOKENS = 3000
    KB_RELEVANCE_SCORE = 0.62
    MULTI_QUERY_RETRIEVAL = True

    # LangSmith（在 .env 中设置下列变量即可自动生效）
    # LANGCHAIN_TRACING_V2=true
    # LANGCHAIN_API_KEY=ls_...
    # LANGCHAIN_PROJECT=zhishiku-prod

    # ── 运行时配置（model_config.json 覆盖 .env）───────────
    @classmethod
    def load_runtime(cls):
        if not os.path.exists(_RUNTIME_CONFIG_PATH):
            return
        try:
            with open(_RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
            if d.get("api_key"):
                cls.DEEPSEEK_API_KEY  = d["api_key"]
            if d.get("base_url"):
                cls.DEEPSEEK_API_BASE = d["base_url"]
            if d.get("model_name"):
                cls.MODEL_NAME        = d["model_name"]
            if d.get("temperature") is not None:
                cls.TEMPERATURE       = float(d["temperature"])
            if d.get("max_tokens"):
                cls.MAX_TOKENS        = int(d["max_tokens"])
        except Exception as e:
            print(f"Warning: 加载 model_config.json 失败: {e}")

    @classmethod
    def save_runtime(cls, data: dict):
        with open(_RUNTIME_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        cls.load_runtime()

    @classmethod
    def get_runtime(cls) -> dict:
        """返回当前 LLM 配置（API key 脱敏）"""
        raw = {}
        if os.path.exists(_RUNTIME_CONFIG_PATH):
            try:
                with open(_RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except Exception:
                pass
        key = raw.get("api_key") or cls.DEEPSEEK_API_KEY or ""
        masked = ("sk-" + "*" * 16 + key[-4:]) if len(key) > 8 else ("*" * len(key))
        return {
            "provider":    raw.get("provider", "deepseek"),
            "api_key":     masked,
            "has_key":     bool(key),
            "base_url":    raw.get("base_url")    or cls.DEEPSEEK_API_BASE,
            "model_name":  raw.get("model_name")  or cls.MODEL_NAME,
            "temperature": raw.get("temperature", cls.TEMPERATURE),
            "max_tokens":  raw.get("max_tokens",  cls.MAX_TOKENS),
        }

    @classmethod
    def validate(cls):
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY 未设置，请检查 .env 文件或在界面中配置")
        try:
            import dashscope  # noqa: F401
            if not cls.QIANWEN_API_KEY:
                raise ValueError("QIANWEN_API_KEY 未设置，请检查 .env 文件")
        except ImportError:
            pass
        return True


# 启动时立即加载运行时配置
Config.load_runtime()
