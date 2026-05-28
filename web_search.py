import time
from threading import Lock

from langchain_community.utilities import SearxSearchWrapper
from langchain_core.documents import Document

from config import Config


class _CircuitBreaker:
    """连续失败 N 次后暂停 recovery_timeout 秒。"""

    def __init__(self, failure_threshold=3, recovery_timeout=60):
        self._failures   = 0
        self._threshold  = failure_threshold
        self._timeout    = recovery_timeout
        self._opened_at  = None
        self._lock       = Lock()

    @property
    def is_open(self):
        with self._lock:
            if self._opened_at is None:
                return False
            if time.time() - self._opened_at > self._timeout:
                self._opened_at = None
                self._failures  = 0
                return False
            return True

    def record_success(self):
        with self._lock:
            self._failures  = 0
            self._opened_at = None

    def record_failure(self):
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._opened_at = time.time()
                print(f"[WebSearch] 熔断器打开，暂停 {self._timeout}s"
                      f"（连续失败 {self._failures} 次）")


def _make_wrapper() -> SearxSearchWrapper | None:
    """创建 SearxSearchWrapper，连接失败时返回 None。"""
    try:
        wrapper = SearxSearchWrapper(
            searx_host=Config.SEARXNG_URL,
            k=Config.WEB_SEARCH_MAX_RESULTS,
        )
        return wrapper
    except Exception as e:
        print(f"[WebSearch] SearXNG 连接失败（{Config.SEARXNG_URL}）: {e}")
        return None


class WebSearcher:
    _breaker = _CircuitBreaker(failure_threshold=3, recovery_timeout=60)

    def __init__(self):
        self._wrapper = _make_wrapper()
        if self._wrapper:
            print(f"[WebSearch] SearXNG 已连接：{Config.SEARXNG_URL}")
        else:
            print("[WebSearch] SearXNG 不可用，网络搜索已禁用。"
                  "\n  启动方式：docker run -d -p 8080:8080 searxng/searxng")

    def search(self, query: str) -> list[Document]:
        if not self._wrapper:
            return []
        if self._breaker.is_open:
            print("[WebSearch] 熔断器开路，跳过网络搜索")
            return []

        last_exc = None
        for attempt in range(3):
            try:
                results = self._do_search(query)
                self._breaker.record_success()
                return results
            except Exception as e:
                last_exc = e
                wait = 2 ** attempt        # 1s, 2s, 4s
                print(f"[WebSearch] 第 {attempt+1} 次失败: {e}，{wait}s 后重试")
                if attempt < 2:
                    time.sleep(wait)

        self._breaker.record_failure()
        print(f"[WebSearch] 三次重试均失败: {last_exc}")
        return []

    def _do_search(self, query: str) -> list[Document]:
        raw = self._wrapper.results(
            query,
            num_results=Config.WEB_SEARCH_MAX_RESULTS,
            language=Config.SEARXNG_LANGUAGE,
        )
        docs = []
        for r in raw:
            snippet = r.get("snippet") or r.get("content", "")
            title   = r.get("title", "")
            link    = r.get("link") or r.get("url", "")
            docs.append(Document(
                page_content=f"{title}\n{snippet}".strip(),
                metadata={
                    "source":  link,
                    "title":   title,
                    "engines": r.get("engines", []),
                    "type":    "web",
                }
            ))
        return docs
