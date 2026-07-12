#!/usr/bin/env python3
"""
app.py -- chat-style local web UI for the Infor LN trace analyzer.

Three-pane layout: uploaded traces on the left, the raw trace file (with
line numbers) in the middle, and a conversational chat on the right where
you can ask follow-up questions naturally.

100% LOCAL -- talks only to Ollama at http://localhost:11434.

HOW CHAT ANSWERS ARE GROUNDED (why this doesn't just hallucinate)
--------------------------------------------------------------------
On every message, before calling the LLM, the server searches the actual
loaded trace file for terms from your question (reusing the same
regex/keyword extraction as the CLI tool) and only sends the LLM the
matching snippets + your question + recent chat history. So the model is
always answering from real lines in your file, not guessing -- and the
UI can tell the middle viewer to jump to and highlight exactly those lines.

For vague follow-ups ("how much is max limit") that don't contain a new
specific term, the server reuses the keywords from your previous message
so context carries across turns, the same way a person would keep reading
the same area of the file.

SETUP
-----
pip install flask ollama psutil --break-system-packages
ollama pull llama3.2:3b

RUN
---
python app.py
Then open http://127.0.0.1:5000
"""

import os
import sys
import re
import uuid
import secrets
from pathlib import Path
from functools import wraps

from flask import Flask, request, jsonify, render_template, redirect, url_for, session

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infor_ln_analyzer as az

try:
    import ollama
except ImportError:
    print("Missing dependency. Run: pip install ollama --break-system-packages")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

APP_PASSWORD = os.environ.get("APP_PASSWORD")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

# In-memory store: one process, single-user or small-team local tool.
# TRACES[trace_id] = {"filename":, "lines": [...], "total": int, "last_keywords": [...]}
TRACES = {}

DEFAULT_MODEL = "llama3.2:3b"


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if APP_PASSWORD and not session.get("authed"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


@app.route("/login", methods=["GET", "POST"])
def login():
    if not APP_PASSWORD:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        if secrets.compare_digest(request.form.get("password", ""), APP_PASSWORD):
            session["authed"] = True
            return redirect(request.args.get("next") or url_for("index"))
        error = "Incorrect password."
    return f"""
    <html><body style="font-family: sans-serif; max-width: 360px; margin: 80px auto; background:#0f1115; color:#e7e9ee;">
      <h2 style="font-family: monospace;">LN Trace Chat</h2>
      <form method="post">
        <input type="password" name="password" placeholder="Access password" autofocus
               style="width:100%; padding:10px; margin-bottom:10px; background:#181b21; color:#e7e9ee; border:1px solid #2a2f38; border-radius:6px;">
        <button type="submit" style="width:100%; padding:10px; background:#f2a93b; border:none; border-radius:6px; font-weight:600; cursor:pointer;">Enter</button>
      </form>
      {'<p style="color:#f2545b;">' + error + '</p>' if error else ''}
    </body></html>
    """


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html", show_logout=bool(APP_PASSWORD))


@app.route("/api/models")
@login_required
def api_models():
    try:
        resp = ollama.list()
        models = [m.get("model") or m.get("name") for m in resp.get("models", [])]
        return jsonify({"ok": True, "models": models})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "models": []})


@app.route("/api/upload", methods=["POST"])
@login_required
def api_upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "No file provided."}), 400

    trace_id = uuid.uuid4().hex
    save_path = UPLOAD_DIR / f"{trace_id}_{f.filename}"
    f.save(str(save_path))

    try:
        lines = az.read_lines(str(save_path))
    except Exception as e:
        return jsonify({"ok": False, "error": f"Could not read file: {e}"}), 400
    finally:
        save_path.unlink(missing_ok=True)  # don't keep the raw upload once parsed into memory

    TRACES[trace_id] = {
        "filename": f.filename,
        "lines": lines,
        "total": len(lines),
        "last_keywords": [],
    }
    return jsonify({"ok": True, "trace_id": trace_id, "filename": f.filename, "total_lines": len(lines)})


@app.route("/api/trace/<trace_id>/lines")
@login_required
def api_trace_lines(trace_id):
    t = TRACES.get(trace_id)
    if not t:
        return jsonify({"ok": False, "error": "Unknown trace_id"}), 404
    start = max(0, int(request.args.get("start", 0)))
    count = min(1000, int(request.args.get("count", 400)))
    end = min(t["total"], start + count)
    lines = [{"n": start + i + 1, "text": t["lines"][start + i].rstrip("\n")} for i in range(end - start)]
    return jsonify({"ok": True, "lines": lines, "start": start, "end": end, "total": t["total"],
                     "filename": t["filename"]})


@app.route("/api/trace/<trace_id>/search")
@login_required
def api_trace_search(trace_id):
    t = TRACES.get(trace_id)
    if not t:
        return jsonify({"ok": False, "error": "Unknown trace_id"}), 404
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"ok": True, "matches": []})
    pattern = re.compile(re.escape(q), re.IGNORECASE)
    matches = []
    for i, line in enumerate(t["lines"]):
        if pattern.search(line):
            matches.append(i + 1)
            if len(matches) >= 200:
                break
    return jsonify({"ok": True, "matches": matches})


CHAT_SYSTEM = (
    "You are a senior Infor LN / Baan ERP support engineer helping someone read a "
    "4GL runtime trace file in a chat. You'll be given snippets retrieved from the "
    "actual trace file that matched their question, plus recent conversation history. "
    "Answer their question conversationally and specifically, citing exact line "
    "numbers from the snippets when relevant (e.g. 'at line 46826'). If the retrieved "
    "snippets don't actually contain enough information to answer, say so plainly "
    "instead of guessing. Keep answers focused and under 120 words unless the "
    "question clearly needs more detail. Do not use markdown headers or bullet-heavy "
    "formatting -- write like you're typing a chat message to a colleague."
)


def extract_keywords_from_message(msg: str):
    q_lower = msg.lower()
    words = re.findall(r"[a-z0-9_.]+", q_lower)
    keywords = [w for w in words if w not in az.STOPWORDS and len(w) >= 3]
    # Real Infor LN identifiers (function names, field names, order/customer
    # IDs) always contain a dot, digit, or underscore -- e.g.
    # "tcibd.dll0010.determine.segment", "S21786802". Plain long English
    # words ("happen", "question") should NOT count as specific terms, or
    # vague follow-ups would wrongly skip context carryover.
    specific = [w for w in keywords if re.search(r"[\d._]", w)]
    return specific


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    data = request.get_json(force=True)
    trace_id = data.get("trace_id")
    message = (data.get("message") or "").strip()
    history = data.get("history", [])  # [{role, content}, ...] from the client
    model = data.get("model") or DEFAULT_MODEL

    t = TRACES.get(trace_id)
    if not t:
        return jsonify({"ok": False, "error": "Unknown trace_id"}), 404
    if not message:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    keywords = extract_keywords_from_message(message)
    used_carryover = False
    if not keywords and t["last_keywords"]:
        keywords = t["last_keywords"]
        used_carryover = True
    if keywords:
        t["last_keywords"] = keywords

    highlight_lines = []
    context_blob = ""
    if keywords:
        raw = az.find_keyword_events(t["lines"], keywords)
        if raw:
            events = az.cluster_events(raw, max_gap=10)
            summaries = az.summarize_events(events)
            incidents = az.dedupe_events(summaries)
            incidents.sort(key=lambda x: -x["occurrences"])
            top = incidents[:5]
            chunks = []
            for inc in top:
                highlight_lines.extend(inc["all_lines"][:5])
                chunks.append(f"--- around line {inc['all_lines'][0]} ---\n" + "\n".join(inc["context"]))
            context_blob = "\n\n".join(chunks)
    else:
        # generic question with no specific terms at all -- fall back to a
        # quick error scan so "why did this fail" style first-messages still work
        incidents = az.get_error_incidents(t["lines"])
        top = sorted(incidents, key=lambda x: -x["occurrences"])[:5]
        chunks = []
        for inc in top:
            highlight_lines.extend(inc["all_lines"][:5])
            chunks.append(f"--- {inc['function']} returned {inc['return_code']}, "
                           f"{inc['occurrences']}x, around line {inc['all_lines'][0]} ---\n"
                           + "\n".join(inc["context"]))
        context_blob = "\n\n".join(chunks)

    if not context_blob:
        context_blob = "(No matching lines were found in the trace for this question.)"

    convo = []
    for turn in history[-8:]:  # keep prompt bounded
        role = "user" if turn.get("role") == "user" else "assistant"
        convo.append({"role": role, "content": turn.get("content", "")})

    user_prompt = (
        (f"(Reusing context from your previous question since this looks like a follow-up.)\n\n"
         if used_carryover else "")
        + f"Question: {message}\n\nRetrieved trace snippets:\n{context_blob}"
    )

    messages = [{"role": "system", "content": CHAT_SYSTEM}] + convo + [{"role": "user", "content": user_prompt}]

    try:
        resp = ollama.chat(model=model, messages=messages, options={"num_predict": 350, "temperature": 0.2})
        reply = resp["message"]["content"].strip()
    except Exception as e:
        return jsonify({"ok": False, "error": f"Ollama error: {e}"}), 500

    # dedupe + cap highlighted lines, sorted
    highlight_lines = sorted(set(highlight_lines))[:20]

    return jsonify({"ok": True, "reply": reply, "highlight_lines": highlight_lines,
                     "jump_to_line": highlight_lines[0] if highlight_lines else None})


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    print("Starting LN Trace Chat UI...")
    if not APP_PASSWORD:
        print("\n\u26a0\ufe0f  APP_PASSWORD not set -- fine for local-only use, "
              "set it before exposing this beyond 127.0.0.1.\n")
    print(f"Open http://{host}:{port} in your browser.")
    app.run(host=host, port=port, debug=False, threaded=True)
