# 17 · Flask Web 服务设计

> 对应主项目 `app.py` 路由结构（非流式部分）

---

## Flask 基础

Flask 是 Python 最流行的轻量级 Web 框架，核心只有两件事：  
**接收 HTTP 请求** → **执行函数** → **返回响应**。

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/hello', methods=['GET'])
def hello():
    name = request.args.get('name', '世界')
    return jsonify({"message": f"你好，{name}！"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

访问 `http://localhost:5000/api/hello?name=张三` 返回：

```json
{"message": "你好，张三！"}
```

---

## 本项目的路由全览

```
GET  /                            → 返回前端 HTML 页面

# 聊天接口
POST /api/chat                    → 非流式问答
POST /api/chat/stream             → SSE 流式问答（主要使用）

# LLM 模型配置
GET  /api/config/llm              → 获取当前模型配置（API Key 脱敏）
POST /api/config/llm              → 保存新模型配置（热切换）
POST /api/config/test             → 临时测试连接（不保存）

# 知识库管理
GET  /api/kb/files                → 列出知识库所有文件
POST /api/kb/upload               → 上传新文档
DELETE /api/kb/file/<path>        → 删除文档
POST /api/kb/rebuild              → 触发异步重建向量库
GET  /api/kb/rebuild/status       → 查询重建进度

# 提示词管理
GET  /api/prompts/versions        → 列出历史版本
POST /api/prompts/save_version    → 存档当前版本
POST /api/prompts/activate/<ver>  → 激活历史版本
GET  /api/prompts/variables       → 读取变量配置
POST /api/prompts/variables       → 更新变量配置
POST /api/prompts/system          → 直接更新 system.txt
POST /api/prompts/reload          → 热重载所有提示词

# 其他
GET  /api/health                  → 健康检查
```

---

## 知识库文件管理

### 上传文件

```python
@app.route('/api/kb/upload', methods=['POST'])
def upload_kb_file():
    f    = request.files['file']
    dest = request.form.get('dest', '').strip().strip('/')  # 子目录（可选）
    ext  = os.path.splitext(f.filename)[1].lower()

    if ext not in {'.txt', '.md', '.pdf'}:
        return jsonify({"error": f"不支持 {ext}"}), 400

    save_dir = os.path.join(_KB_DIR, dest) if dest else _KB_DIR
    os.makedirs(save_dir, exist_ok=True)
    f.save(os.path.join(save_dir, os.path.basename(f.filename)))
    return jsonify({"status": "ok"})
```

### 删除文件（路径安全校验）

```python
@app.route('/api/kb/file/<path:rel_path>', methods=['DELETE'])
def delete_kb_file(rel_path):
    fpath = os.path.join(_KB_DIR, rel_path)
    real  = os.path.realpath(fpath)      # 解析所有 ../.. 符号链接

    # 防目录遍历攻击：确保目标文件在知识库目录内
    if not real.startswith(os.path.realpath(_KB_DIR)):
        return jsonify({"error": "非法路径"}), 403

    os.remove(fpath)
    return jsonify({"status": "ok"})
```

**目录遍历攻击：** 攻击者可能传入 `../../.env` 这样的路径，  
`realpath()` 解析后与知识库目录对比，有效阻止越权访问。

---

## 后台重建：异步 + 状态轮询

向量库重建可能需要几十秒，不能阻塞 HTTP 请求。  
本项目用后台线程 + 状态字典实现异步重建：

```python
_rebuild_state = {"status": "idle", "chunks": 0, "message": ""}
_rebuild_lock  = threading.Lock()

def _do_rebuild():
    global kb
    with _rebuild_lock:    # 防止并发重建
        _rebuild_state.update({"status": "running", "message": "正在重建..."})
        try:
            kb.build_vector_store()
            n = len(list(kb.vector_store.docstore._dict.values()))
            _rebuild_state.update({"status": "done", "chunks": n,
                                    "message": f"完成，共 {n} 个片段"})
        except Exception as e:
            _rebuild_state.update({"status": "error", "message": str(e)})

@app.route('/api/kb/rebuild', methods=['POST'])
def rebuild_kb():
    if _rebuild_state["status"] == "running":
        return jsonify({"status": "already_running"})
    threading.Thread(target=_do_rebuild, daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/api/kb/rebuild/status', methods=['GET'])
def rebuild_status():
    return jsonify(_rebuild_state)
```

前端定期轮询 `/api/kb/rebuild/status`，直到 `status == "done"` 或 `"error"`。

---

## 模型热切换

```python
@app.route('/api/config/llm', methods=['POST'])
def set_llm_config():
    global llm
    data = request.get_json()

    # API Key 脱敏处理：如果用户没改动（显示的是 sk-***...），保留原来的 key
    if data['api_key'].startswith('sk-') and '*' in data['api_key']:
        with open('model_config.json') as f:
            data['api_key'] = json.load(f).get('api_key', Config.DEEPSEEK_API_KEY)

    Config.save_runtime(data)    # 写入 model_config.json
    llm = LLMClient()            # 重新实例化（使用新配置）
    return jsonify({"status": "ok", "model": Config.MODEL_NAME})
```

---

## 连接测试端点

用户修改配置后，点击"测试"可以临时验证新配置，不影响当前使用中的模型：

```python
@app.route('/api/config/test', methods=['POST'])
def test_llm_config():
    data = request.get_json()
    test_llm = ChatOpenAI(
        api_key=data['api_key'],
        base_url=data['base_url'],
        model=data['model_name'],
        temperature=0, max_tokens=32    # 最小开销的测试调用
    )
    reply = (test_llm | StrOutputParser()).invoke("用一句话介绍你自己")
    return jsonify({"status": "ok", "reply": reply[:120]})
```

---

## 动手练习

1. 用 `curl` 或 Postman 调用 `/api/kb/files`，查看知识库文件列表
2. 上传一个测试文档，触发重建，轮询状态直到完成
3. 尝试传入 `../../config.py` 到删除接口，观察安全校验的效果
4. 思考：为什么 `_rebuild_lock` 用 `threading.Lock()` 而不是直接判断 `status == "running"`？
