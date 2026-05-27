import json
import uuid

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from chain import query_chat, query_rag, stream_rag
from config import DEFAULT_ALPHA, DEFAULT_TOP_K

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K
    alpha: float = DEFAULT_ALPHA


class ChatRequest(BaseModel):
    question: str
    session_id: str = ""
    top_k: int = DEFAULT_TOP_K
    alpha: float = DEFAULT_ALPHA


@app.get("/", response_class=HTMLResponse)
def home_page() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Gov Budget RAG</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--primary:#2563eb;--primary-hover:#1d4ed8;--bg:#f8fafc;--card:#fff;--border:#e2e8f0;--text:#1e293b;--muted:#64748b;--radius:10px}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column}
.header{background:var(--primary);padding:0.7rem 1.5rem;display:flex;align-items:center;justify-content:space-between}
.header h1{color:#fff;font-size:1.05rem;font-weight:600;letter-spacing:-0.01em}
.tabs{display:flex;gap:0;margin:0;padding:0;background:var(--primary)}
.tab-btn{background:rgba(255,255,255,0.15);color:rgba(255,255,255,0.85);border:none;padding:0.5rem 1.2rem;cursor:pointer;font-size:0.82rem;font-weight:500;transition:background 0.2s}
.tab-btn:hover{background:rgba(255,255,255,0.25)}
.tab-btn.active{background:rgba(255,255,255,0.95);color:var(--primary);font-weight:600}
.tab-body{flex:1;overflow-y:auto;position:relative}
.tab-content{display:none;height:100%;flex-direction:column}
.tab-content.active{display:flex}

/* Chat tab */
.chat-area{flex:1;overflow-y:auto;padding:1rem}
.msg{max-width:78%;margin-bottom:0.65rem;padding:0.55rem 0.9rem;border-radius:14px;line-height:1.55;font-size:0.88rem;white-space:pre-wrap;word-wrap:break-word}
.msg.user{background:var(--primary);color:#fff;margin-left:auto;border-bottom-right-radius:4px}
.msg.bot{background:var(--card);color:var(--text);border:1px solid var(--border);border-bottom-left-radius:4px}
.sources{font-size:0.72rem;color:var(--muted);margin-top:0.2rem;padding:0.1rem 0}
.sources b{color:var(--primary)}
.input-bar{padding:0.65rem 1rem;background:var(--card);border-top:1px solid var(--border);display:flex;gap:0.5rem}
.input-bar input{flex:1;padding:0.55rem 0.9rem;border:1px solid var(--border);border-radius:8px;font-size:0.88rem;outline:none;transition:border 0.2s}
.input-bar input:focus{border-color:var(--primary)}
.input-bar button{background:var(--primary);color:#fff;border:none;padding:0.55rem 1.1rem;border-radius:8px;cursor:pointer;font-size:0.88rem;font-weight:500;transition:background 0.2s}
.input-bar button:hover{background:var(--primary-hover)}
.input-bar button:disabled{background:#94a3b8;cursor:not-allowed}
.chat-actions{display:flex;align-items:center;gap:0.5rem}
.chat-actions button{background:transparent;color:#fff;border:1px solid rgba(255,255,255,0.35);border-radius:4px;padding:0.25rem 0.7rem;cursor:pointer;font-size:0.78rem}
.chat-actions button:hover{background:rgba(255,255,255,0.2)}

/* Simple & Debug tabs */
.simple-area{flex:1;overflow-y:auto;padding:1.5rem}
.result-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem;margin-bottom:1rem}
.result-card h3{font-size:0.95rem;margin-bottom:0.6rem;color:var(--primary)}
.result-card pre{white-space:pre-wrap;word-wrap:break-word;font-size:0.85rem;line-height:1.55;background:#f1f5f9;padding:0.8rem;border-radius:6px}
.source-tag{display:inline-block;background:#dbeafe;color:var(--primary);padding:0.15rem 0.5rem;border-radius:4px;font-size:0.72rem;font-weight:500;margin:0.15rem}
.score-badge{display:inline-block;background:#f0fdf4;color:#166534;padding:0.15rem 0.5rem;border-radius:4px;font-size:0.72rem;font-weight:500;margin:0.15rem}
.debug-row{display:flex;justify-content:space-between;align-items:flex-start;padding:0.6rem 0;border-bottom:1px solid var(--border)}
.debug-row:last-child{border-bottom:none}
.debug-doc{flex:1;font-size:0.82rem;line-height:1.5}
.debug-meta{font-size:0.75rem;color:var(--muted);min-width:140px;text-align:right}
.empty-msg{text-align:center;color:var(--muted);padding:3rem 1rem;font-size:0.9rem}

/* Params */
.params{display:none;padding:0.6rem 1rem;background:#f1f5f9;border-bottom:1px solid var(--border);gap:0.8rem;align-items:center;flex-wrap:wrap}
.params.show{display:flex}
.params label{font-size:0.78rem;color:var(--muted);font-weight:500}
.params input[type=number]{width:70px;padding:0.3rem 0.5rem;border:1px solid var(--border);border-radius:4px;font-size:0.8rem}
</style>
</head>
<body>
<div class="header">
  <h1>Gov Budget RAG</h1>
  <div class="chat-actions">
    <button onclick="toggleParams()">Params</button>
    <button onclick="newChat()">New Chat</button>
  </div>
</div>
<div class="tabs" id="tabs">
  <button class="tab-btn active" data-tab="chat" onclick="switchTab('chat')">Stream Chat</button>
  <button class="tab-btn" data-tab="simple" onclick="switchTab('simple')">Simple Query</button>
  <button class="tab-btn" data-tab="debug" onclick="switchTab('debug')">Debug Search</button>
</div>
<div class="params" id="params">
  <label>top_k <input type="number" id="paramTopK" value="5" min="1" max="50"/></label>
  <label>alpha <input type="number" id="paramAlpha" value="0.6" min="0" max="1" step="0.1"/></label>
</div>

<!-- CHAT TAB -->
<div class="tab-body">
  <div class="tab-content active" id="tab-chat">
    <div class="chat-area" id="chatArea"></div>
    <div class="input-bar">
      <input type="text" id="chatInput" placeholder="Ask about government budgets..."/>
      <button id="chatSend" onclick="sendChat()">Send</button>
    </div>
  </div>
  <div class="tab-content" id="tab-simple">
    <div class="simple-area" id="simpleArea">
      <div class="empty-msg" id="simpleEmpty">Submit a query to get an answer with sources.</div>
    </div>
    <div class="input-bar">
      <input type="text" id="simpleInput" placeholder="Ask a question..."/>
      <button id="simpleSend" onclick="sendSimple()">Ask</button>
    </div>
  </div>
  <div class="tab-content" id="tab-debug">
    <div class="simple-area" id="debugArea">
      <div class="empty-msg" id="debugEmpty">Submit a query to see raw search results with scores.</div>
    </div>
    <div class="input-bar">
      <input type="text" id="debugInput" placeholder="Search query..."/>
      <button id="debugSend" onclick="sendDebug()">Search</button>
    </div>
  </div>
</div>

<script>
let sessionId = crypto.randomUUID();
let chatBusy = false;

const chatArea = document.getElementById('chatArea');
const chatInput = document.getElementById('chatInput');
const chatSend = document.getElementById('chatSend');

chatInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendChat(); });
document.getElementById('simpleInput').addEventListener('keydown', e => { if (e.key === 'Enter') sendSimple(); });
document.getElementById('debugInput').addEventListener('keydown', e => { if (e.key === 'Enter') sendDebug(); });

function getTopK() { return parseInt(document.getElementById('paramTopK').value) || 5; }
function getAlpha() { return parseFloat(document.getElementById('paramAlpha').value) || 0.6; }

function toggleParams() {
  document.getElementById('params').classList.toggle('show');
}

function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.toggle('active', c.id === 'tab-' + tab));
}

function newChat() {
  sessionId = crypto.randomUUID();
  chatArea.innerHTML = '';
}

function addMsg(text, role) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
  return div;
}

function addSourcesEl(sources) {
  if (!sources || !sources.length) return;
  const div = document.createElement('div');
  div.className = 'sources';
  div.innerHTML = '<b>Sources:</b> ' + sources.map(s =>
    `<span class="source-tag">${s.source || 'unknown'} #${s.id || ''}</span>`
  ).join(' ');
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function setChatBusy(val) {
  chatBusy = val;
  chatSend.disabled = val;
  chatInput.disabled = val;
}

// ── STREAM CHAT ──
async function sendChat() {
  const query = chatInput.value.trim();
  if (!query || chatBusy) return;
  chatInput.value = '';
  setChatBusy(true);
  addMsg(query, 'user');
  const botMsg = addMsg('', 'bot');

  try {
    const res = await fetch('/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: query, session_id: sessionId, top_k: getTopK(), alpha: getAlpha()})
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, {stream: true});
      for (const line of chunk.split('\n')) {
        if (!line.trim()) continue;
        try {
          const event = JSON.parse(line);
          if (event.type === 'sources') addSourcesEl(event.data);
          else if (event.type === 'token') { fullText += event.text; botMsg.textContent = fullText; chatArea.scrollTop = chatArea.scrollHeight; }
          else if (event.type === 'done') break;
        } catch(e) { fullText += line; botMsg.textContent = fullText; }
      }
    }
    if (!fullText) botMsg.textContent = 'No answer received.';
  } catch(err) {
    botMsg.textContent = 'Error: ' + err.message;
  } finally { setChatBusy(false); chatInput.focus(); }
}

// ── SIMPLE RAG QUERY ──
async function sendSimple() {
  const query = document.getElementById('simpleInput').value.trim();
  if (!query) return;
  document.getElementById('simpleInput').value = '';
  const area = document.getElementById('simpleArea');
  const empty = document.getElementById('simpleEmpty');
  if (empty) empty.remove();

  area.innerHTML += '<div class="result-card"><h3>Q: ' + escHtml(query) + '</h3><div id="simpleLoading" style="color:var(--muted);font-style:italic">Thinking...</div></div>';

  try {
    const res = await fetch('/rag', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, top_k: getTopK(), alpha: getAlpha()})
    });
    const data = await res.json();
    const card = area.lastElementChild;
    card.innerHTML = '<h3>Q: ' + escHtml(query) + '</h3>' +
      '<pre>' + escHtml(data.answer || 'No answer') + '</pre>' +
      '<div>' + (data.sources || []).map(s =>
        '<span class="source-tag">' + escHtml(s.source || 'unknown') + ' #' + escHtml(String(s.id || '')) + '</span>'
      ).join(' ') + '</div>';
  } catch(err) {
    area.lastElementChild.innerHTML = '<h3>Error</h3><pre>' + escHtml(err.message) + '</pre>';
  }
}

// ── DEBUG SEARCH ──
async function sendDebug() {
  const query = document.getElementById('debugInput').value.trim();
  if (!query) return;
  document.getElementById('debugInput').value = '';
  const area = document.getElementById('debugArea');
  const empty = document.getElementById('debugEmpty');
  if (empty) empty.remove();

  area.innerHTML += '<div class="result-card"><h3>Search: ' + escHtml(query) + '</h3><div style="color:var(--muted);font-style:italic">Searching...</div></div>';

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, top_k: getTopK(), alpha: getAlpha()})
    });
    const data = await res.json();
    const results = data.results || [];
    const card = area.lastElementChild;
    if (!results.length) {
      card.innerHTML = '<h3>Search: ' + escHtml(query) + '</h3><div class="empty-msg">No results found.</div>';
      return;
    }
    let html = '<h3>Search: ' + escHtml(query) + ' &mdash; ' + results.length + ' results</h3>';
    results.forEach((r, i) => {
      const src = (r.metadata && r.metadata.source) || 'unknown';
      html += '<div class="debug-row">' +
        '<div class="debug-doc"><strong>' + (i+1) + '.</strong> ' + escHtml(String(r.document || '').substring(0, 300)) + (String(r.document || '').length > 300 ? '...' : '') + '</div>' +
        '<div class="debug-meta"><span class="source-tag">' + escHtml(src) + '</span>' +
        '<span class="score-badge">hybrid: ' + Number(r.score).toFixed(3) + '</span>' +
        '<span class="score-badge">sem: ' + Number(r.semantic_score || 0).toFixed(3) + '</span>' +
        '<span class="score-badge">kw: ' + Number(r.keyword_score || 0).toFixed(3) + '</span></div></div>';
    });
    card.innerHTML = html;
  } catch(err) {
    area.lastElementChild.innerHTML = '<h3>Error</h3><pre>' + escHtml(err.message) + '</pre>';
  }
}

function escHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}
</script>
</body>
</html>
"""


@app.post("/rag")
async def rag_endpoint(payload: QueryRequest) -> dict:
    result = await run_in_threadpool(
        query_rag, payload.query, payload.top_k, payload.alpha
    )
    return {
        "ok": True,
        "query": payload.query,
        "answer": result["answer"],
        "sources": result["sources"],
    }


@app.post("/chat")
async def chat_endpoint(payload: ChatRequest) -> dict:
    result = await run_in_threadpool(
        query_chat, payload.question, payload.session_id, payload.top_k, payload.alpha
    )
    return {
        "ok": True,
        "answer": result["answer"],
        "sources": result["sources"],
        "session_id": result["session_id"],
    }


@app.post("/stream")
async def stream_endpoint(payload: ChatRequest) -> StreamingResponse:
    def generate():
        for line in stream_rag(payload.question, payload.top_k, payload.alpha):
            yield line

    return StreamingResponse(generate(), media_type="text/x-ndjson")


@app.post("/query")
async def query_endpoint(payload: QueryRequest) -> dict:
    from retrieval import hybrid_search

    results = await run_in_threadpool(
        hybrid_search, payload.query, payload.top_k, payload.alpha
    )
    return {"ok": True, "query": payload.query, "results": results}