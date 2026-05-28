import sys
import os
import json
import shutil
import threading
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, Response
import re
import markdown

from langchain_core.messages import HumanMessage, AIMessage

from config import Config
from knowledge_base import KnowledgeBase
from llm_client import LLMClient
from web_search import WebSearcher

# tiktoken 精确 token 计数（不可用时降级到字符数估算）
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    print("Warning: tiktoken 未安装，使用字符数估算 token 数")
    def _count_tokens(text: str) -> int:
        return len(text) // 2   # 中文约 2 字/token

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

print("=" * 60)
print("   LangChain + DeepSeek + 本地知识库问答系统 (Web版)")
print("=" * 60)

try:
    Config.validate()
except ValueError as e:
    print(f"配置错误: {e}")
    sys.exit(1)

kb = KnowledgeBase()
llm = LLMClient()
web_searcher = WebSearcher() if Config.WEB_SEARCH_ENABLED else None

if not kb.load_vector_store():
    print("未找到本地向量数据库，开始构建...")
    kb.build_vector_store()

print("系统已就绪！")


# ── Markdown / 公式渲染 ─────────────────────────────────────
def render_markdown(text):
    if not text:
        return ""

    code_blocks = []
    def save_code(m):
        code_blocks.append(m.group(0))
        return f'\x00CODE{len(code_blocks)-1}\x00'
    text = re.sub(r'```[\s\S]*?```', save_code, text)

    block_math = []
    def save_block(m):
        block_math.append(m.group(1).strip())
        return f'\x00BMATH{len(block_math)-1}\x00'
    text = re.sub(r'\$\$(.+?)\$\$', save_block, text, flags=re.DOTALL)

    inline_math = []
    def save_inline(m):
        inline_math.append(m.group(1).strip())
        return f'\x00IMATH{len(inline_math)-1}\x00'
    text = re.sub(r'\$(.+?)\$', save_inline, text)

    html = markdown.markdown(text, extensions=['extra', 'fenced_code', 'codehilite'])

    for i, f in enumerate(inline_math):
        html = html.replace(f'\x00IMATH{i}\x00',
            f'<span class="math-inline" data-latex="{_esc(f)}">\\({f}\\)</span>')

    for i, f in enumerate(block_math):
        ph = f'\x00BMATH{i}\x00'
        rep = f'<div class="math-block" data-latex="{_esc(f)}">\\[{f}\\]</div>'
        html = html.replace(f'<p>{ph}</p>', rep).replace(ph, rep)

    for i, b in enumerate(code_blocks):
        html = html.replace(f'\x00CODE{i}\x00', b)

    return html


def _esc(s):
    return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;')


# 过滤模型固有的"根据知识库内容"开头套话
_BAD_OPENER = re.compile(
    r'^[\s\n]*(根据(提供的|以上|本|这些)?(知识库|资料|搜索结果|参考)(内容|信息|数据)?'
    r'[，,。]?\s*|知识库(中)?(提到|说明|显示|记录)[，,]?\s*)',
    re.IGNORECASE
)

def _clean(text: str) -> str:
    cleaned = _BAD_OPENER.sub('', text, count=1).lstrip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


# ── 参考资料提取 ────────────────────────────────────────────
def extract_references(context):
    return [{
        "index": i,
        "source": os.path.basename(d.metadata.get("source", "unknown")),
        "content": d.page_content,
        "content_html": render_markdown(d.page_content)
    } for i, d in enumerate(context, 1)]


def extract_web_references(web_context):
    return [{
        "index": i,
        "title": d.metadata.get("title", ""),
        "source": d.metadata.get("source", ""),
        "content": d.page_content,
        "content_html": render_markdown(d.page_content)
    } for i, d in enumerate(web_context, 1)]


def parse_history(history_data):
    """
    前端传来的 [{role, content}] → LangChain 消息对象列表。
    按 token 数从最新一轮往前截取，确保总量不超过 MAX_HISTORY_TOKENS。
    """
    items = list(reversed(history_data or []))
    total_tokens = 0
    selected = []
    for item in items:
        role    = item.get('role', '')
        content = item.get('content', '')
        t = _count_tokens(content)
        if total_tokens + t > Config.MAX_HISTORY_TOKENS:
            break
        selected.insert(0, (role, content))
        total_tokens += t

    msgs = []
    for role, content in selected:
        if role == 'user':
            msgs.append(HumanMessage(content=content))
        elif role == 'assistant':
            msgs.append(AIMessage(content=content))
    return msgs


_DIALOGUE_REF = re.compile(
    r'(上一个|上一条|上一轮|上面|刚才|之前(说|问|讲|提到?)?|你说|你讲|你提到|'
    r'你回答|你的回答|你刚|你之前|我(刚才|之前)(问|说)|我们(聊|说|讨论)|'
    r'对话(内容|记录)?|历史(记录)?|前面(说|提)|你(确定|确认|肯定)|'
    r'对吗|正确吗|有误|错了吗|再(说|讲|解释)一?下?|重复一下|'
    r'换[个种]方式|继续|你上面|你前面|解释一下刚|你说的)',
    re.IGNORECASE
)

_SHORT_PRONOUN = re.compile(r'(这个|那个|它|这些|那些|此)')

def _is_dialogue_ref(query: str, history: list) -> bool:
    """判断是否是对当前对话本身的引用，只有存在历史记录时才判定。"""
    if not history:
        return False
    if _DIALOGUE_REF.search(query):
        return True
    # 短问题（≤15字）+ 指代词 → 大概率引用历史
    if len(query) <= 15 and _SHORT_PRONOUN.search(query):
        return True
    return False


def _parallel_search(query):
    """KB + 网络并行检索，返回 (context, web_context)。"""
    if web_searcher:
        with ThreadPoolExecutor(max_workers=2) as ex:
            kb_f  = ex.submit(kb.search, query, Config.TOP_K_RESULTS, llm.llm)
            web_f = ex.submit(web_searcher.search, query)
            return kb_f.result(timeout=20), web_f.result(timeout=15)
    return kb.search(query, llm=llm.llm), []


# ── SSE 工具 ────────────────────────────────────────────────
def _sse(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ── 配置路由 ────────────────────────────────────────────────
@app.route('/api/config/llm', methods=['GET'])
def get_llm_config():
    return jsonify(Config.get_runtime())


@app.route('/api/prompts/reload', methods=['POST'])
def reload_prompts_api():
    """热重载提示词文件，无需重启服务。"""
    global llm
    try:
        import prompts as _p
        _p.reload_prompts()
        llm = LLMClient()   # 用新提示词重建链
        return jsonify({"status": "ok", "message": "提示词已重载"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/config/llm', methods=['POST'])
def set_llm_config():
    global llm
    data = request.get_json()
    required = ['api_key', 'base_url', 'model_name']
    if not all(data.get(k, '').strip() for k in required):
        return jsonify({"error": "api_key、base_url、model_name 均为必填项"}), 400

    # 若 api_key 是脱敏占位符（用户未改动），保留原有 key
    if data['api_key'].startswith('sk-') and '*' in data['api_key']:
        existing = {}
        import os as _os
        if _os.path.exists(os.path.join(os.path.dirname(__file__), 'model_config.json')):
            import json as _json
            with open(os.path.join(os.path.dirname(__file__), 'model_config.json'), 'r') as f:
                existing = _json.load(f)
        data['api_key'] = existing.get('api_key', Config.DEEPSEEK_API_KEY or '')

    try:
        Config.save_runtime(data)
        llm = LLMClient()     # 热重载 LLM 客户端
        return jsonify({"status": "ok", "model": Config.MODEL_NAME})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/config/test', methods=['POST'])
def test_llm_config():
    """用表单里的临时参数测试连通性，不保存。"""
    data = request.get_json()
    api_key   = data.get('api_key', '').strip()
    base_url  = data.get('base_url', '').strip()
    model_name = data.get('model_name', '').strip()

    # 脱敏 key → 用当前已保存的 key
    if not api_key or (api_key.startswith('sk-') and '*' in api_key):
        api_key = Config.DEEPSEEK_API_KEY or ''

    if not all([api_key, base_url, model_name]):
        return jsonify({"error": "参数不完整"}), 400

    try:
        from langchain_openai import ChatOpenAI as _Chat
        from langchain_core.output_parsers import StrOutputParser as _Parser
        test_llm = _Chat(api_key=api_key, base_url=base_url,
                         model=model_name, temperature=0, max_tokens=32)
        reply = (test_llm | _Parser()).invoke("用一句话介绍你自己")
        return jsonify({"status": "ok", "reply": reply[:120]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 提示词版本管理 ──────────────────────────────────────────
_PROMPT_DIR     = os.path.join(os.path.dirname(__file__), "prompt_templates")
_VERSIONS_DIR   = os.path.join(_PROMPT_DIR, "versions")
_SYSTEM_TXT     = os.path.join(_PROMPT_DIR, "system.txt")
_VARIABLES_JSON = os.path.join(_PROMPT_DIR, "variables.json")

os.makedirs(_VERSIONS_DIR, exist_ok=True)


@app.route('/api/prompts/versions', methods=['GET'])
def list_prompt_versions():
    """列出所有历史版本及当前 system.txt 内容。"""
    versions = []
    if os.path.exists(_VERSIONS_DIR):
        for f in sorted(os.listdir(_VERSIONS_DIR), reverse=True):
            if f.endswith('.txt'):
                versions.append(f)
    current = ""
    if os.path.exists(_SYSTEM_TXT):
        with open(_SYSTEM_TXT, 'r', encoding='utf-8') as f:
            current = f.read()
    return jsonify({"versions": versions, "current": current})


@app.route('/api/prompts/save_version', methods=['POST'])
def save_prompt_version():
    """将当前 system.txt 存档为带时间戳的版本。"""
    if not os.path.exists(_SYSTEM_TXT):
        return jsonify({"error": "system.txt 不存在"}), 404
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(_VERSIONS_DIR, f"system_{ts}.txt")
    shutil.copy2(_SYSTEM_TXT, dest)
    return jsonify({"status": "ok", "version": f"system_{ts}.txt"})


@app.route('/api/prompts/activate/<version>', methods=['POST'])
def activate_prompt_version(version):
    """将指定版本激活为当前 system.txt 并热重载。"""
    global llm
    src = os.path.join(_VERSIONS_DIR, version)
    if not os.path.exists(src):
        return jsonify({"error": f"版本不存在: {version}"}), 404
    shutil.copy2(src, _SYSTEM_TXT)
    try:
        import prompts as _p
        _p.reload_prompts()
        llm = LLMClient()
        return jsonify({"status": "ok", "activated": version})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/prompts/variables', methods=['GET'])
def get_prompt_variables():
    """读取 variables.json。"""
    if not os.path.exists(_VARIABLES_JSON):
        return jsonify({"system_vars": {}, "user_vars": {}})
    with open(_VARIABLES_JSON, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))


@app.route('/api/prompts/variables', methods=['POST'])
def set_prompt_variables():
    """更新 variables.json 并热重载提示词。"""
    global llm
    data = request.get_json()
    with open(_VARIABLES_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    try:
        import prompts as _p
        _p.reload_prompts()
        llm = LLMClient()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/prompts/system', methods=['POST'])
def update_system_prompt():
    """直接更新 system.txt 内容（不存档）并热重载。"""
    global llm
    data = request.get_json()
    content = data.get('content', '').strip()
    if not content:
        return jsonify({"error": "内容不能为空"}), 400
    with open(_SYSTEM_TXT, 'w', encoding='utf-8') as f:
        f.write(content)
    try:
        import prompts as _p
        _p.reload_prompts()
        llm = LLMClient()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 健康检查 ────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({"status": "ok"})


# ── 知识库管理 ──────────────────────────────────────────────
_KB_DIR = Config.KNOWLEDGE_BASE_DIR
_ALLOWED_EXT = {'.txt', '.md', '.pdf'}
_rebuild_state = {"status": "idle", "chunks": 0, "message": ""}
_rebuild_lock  = threading.Lock()


@app.route('/api/kb/files', methods=['GET'])
def list_kb_files():
    files = []
    for root, dirs, fnames in os.walk(_KB_DIR):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for fname in fnames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _ALLOWED_EXT:
                continue
            fpath = os.path.join(root, fname)
            rel   = os.path.relpath(fpath, _KB_DIR).replace('\\', '/')
            stat  = os.stat(fpath)
            files.append({
                "name":     fname,
                "path":     rel,
                "size":     stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
            })
    files.sort(key=lambda f: f['modified'], reverse=True)
    return jsonify({"files": files, "rebuild_status": _rebuild_state["status"]})


@app.route('/api/kb/upload', methods=['POST'])
def upload_kb_file():
    if 'file' not in request.files:
        return jsonify({"error": "未收到文件"}), 400
    f    = request.files['file']
    dest = request.form.get('dest', '').strip().strip('/')
    ext  = os.path.splitext(f.filename)[1].lower()
    if ext not in _ALLOWED_EXT:
        return jsonify({"error": f"不支持的文件类型 {ext}，仅支持 .txt .md .pdf"}), 400

    save_dir = os.path.join(_KB_DIR, dest) if dest else _KB_DIR
    os.makedirs(save_dir, exist_ok=True)
    safe_name = os.path.basename(f.filename)
    f.save(os.path.join(save_dir, safe_name))
    return jsonify({"status": "ok", "saved": os.path.join(dest, safe_name).replace('\\', '/')})


@app.route('/api/kb/file/<path:rel_path>', methods=['DELETE'])
def delete_kb_file(rel_path):
    fpath = os.path.join(_KB_DIR, rel_path)
    real  = os.path.realpath(fpath)
    if not real.startswith(os.path.realpath(_KB_DIR)):
        return jsonify({"error": "非法路径"}), 403
    if not os.path.exists(fpath):
        return jsonify({"error": "文件不存在"}), 404
    os.remove(fpath)
    return jsonify({"status": "ok"})


def _do_rebuild():
    global kb
    with _rebuild_lock:
        _rebuild_state.update({"status": "running", "message": "正在重建向量库..."})
        try:
            kb.build_vector_store()
            n = len(list(kb.vector_store.docstore._dict.values())) if kb.vector_store else 0
            _rebuild_state.update({"status": "done", "chunks": n, "message": f"完成，共 {n} 个片段"})
        except Exception as e:
            _rebuild_state.update({"status": "error", "message": str(e)})


@app.route('/api/kb/rebuild', methods=['POST'])
def rebuild_kb():
    if _rebuild_state["status"] == "running":
        return jsonify({"status": "already_running"})
    _rebuild_state.update({"status": "idle", "chunks": 0, "message": ""})
    threading.Thread(target=_do_rebuild, daemon=True).start()
    return jsonify({"status": "started"})


@app.route('/api/kb/rebuild/status', methods=['GET'])
def rebuild_status():
    return jsonify(_rebuild_state)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data  = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({"error": "请输入问题"}), 400

    try:
        history = parse_history(data.get('history', []))
        context, web_context = _parallel_search(query)
        answer = llm.generate_with_dual_context(query, context, web_context, history)
        return jsonify({
            "query":          query,
            "answer":         answer,
            "answer_html":    render_markdown(answer),
            "references":     extract_references(context)     if context     else [],
            "web_references": extract_web_references(web_context) if web_context else []
        })
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": f"处理请求时出错：{e}"}), 500


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    data  = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({"error": "请输入问题"}), 400

    try:
        history = parse_history(data.get('history', []))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    def generate():
        # ① 对话引用类问题（"上一个问题是什么"）→ 直接走记忆，跳过检索
        if _is_dialogue_ref(query, history):
            yield _sse({'type': 'refs', 'references': [], 'web_references': []})
            full = []
            try:
                for chunk in llm.stream(query, None, None, history):
                    if chunk:
                        full.append(chunk)
                        yield _sse({'type': 'token', 'content': chunk})
            except Exception as e:
                yield _sse({'type': 'error', 'content': str(e)})
            yield _sse({'type': 'done', 'answer_html': render_markdown(''.join(full))})
            yield "data: [DONE]\n\n"
            return

        # ② 常规问题：先检索再回答
        yield _sse({'type': 'status', 'content': '正在检索知识库和互联网...'})

        try:
            context, web_context = _parallel_search(query)
            references     = extract_references(context)          if context     else []
            web_references = extract_web_references(web_context)  if web_context else []
        except Exception as e:
            yield _sse({'type': 'error', 'content': f'检索失败：{e}'})
            return

        # ② 把参考资料推给前端（检索已完成，不用等 LLM）
        yield _sse({'type': 'refs', 'references': references, 'web_references': web_references})

        # ③ 逐 token 流式输出（缓冲开头以过滤套话）
        full = []
        buf = ""
        buf_flushed = False
        BUF_SIZE = 40  # 足够覆盖最长的"根据..."开头

        try:
            for chunk in llm.stream(query, context or None, web_context or None, history):
                if not chunk:
                    continue
                if not buf_flushed:
                    buf += chunk
                    if len(buf) >= BUF_SIZE:
                        buf = _clean(buf)
                        buf_flushed = True
                        full.append(buf)
                        yield _sse({'type': 'token', 'content': buf})
                        buf = ""
                else:
                    full.append(chunk)
                    yield _sse({'type': 'token', 'content': chunk})
        except Exception as e:
            yield _sse({'type': 'error', 'content': str(e)})

        # 剩余缓冲（回答很短时 buf 未被 flush）
        if buf:
            buf = _clean(buf)
            full.append(buf)
            yield _sse({'type': 'token', 'content': buf})

        # ④ 发送服务端渲染好的 Markdown HTML
        yield _sse({'type': 'done', 'answer_html': render_markdown(''.join(full))})
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
