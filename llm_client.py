from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from config import Config
from prompts import KB_ONLY, DUAL_SOURCE, WEB_ONLY, GENERAL


class LLMClient:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_API_BASE,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        parser = StrOutputParser()

        self.kb_chain      = KB_ONLY      | self.llm | parser
        self.dual_chain    = DUAL_SOURCE  | self.llm | parser
        self.web_chain     = WEB_ONLY     | self.llm | parser
        self.general_chain = GENERAL      | self.llm | parser

    def _fmt_kb(self, docs):
        return "\n\n".join(d.page_content for d in docs)

    def _fmt_web(self, docs):
        return "\n\n".join(
            f"[{d.metadata.get('title', '')}]\n{d.page_content}" for d in docs
        )

    def _pick(self, kb, web):
        if kb and web:
            return self.dual_chain, {"kb_context": self._fmt_kb(kb),
                                     "web_context": self._fmt_web(web)}
        if kb:
            return self.kb_chain,   {"context": self._fmt_kb(kb)}
        if web:
            return self.web_chain,  {"web_context": self._fmt_web(web)}
        return self.general_chain,  {}

    def generate_with_dual_context(self, query, kb_context, web_context, history=None):
        chain, inputs = self._pick(kb_context, web_context)
        inputs["query"]   = query
        inputs["history"] = history or []
        return chain.invoke(inputs)

    def stream(self, query, kb_context=None, web_context=None, history=None):
        chain, inputs = self._pick(kb_context, web_context)
        inputs["query"]   = query
        inputs["history"] = history or []
        return chain.stream(inputs)

    # 向后兼容
    def generate_response(self, query):
        return self.general_chain.invoke({"query": query, "history": []})

    def generate_with_context(self, query, context):
        return self.kb_chain.invoke({"context": self._fmt_kb(context),
                                     "query": query, "history": []})
