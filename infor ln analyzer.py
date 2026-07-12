#!/usr/bin/env python3
"""
infor_ln_analyzer.py
Analyzes Infor LN / Baan 4GL runtime trace files (the "B:<line>:::(<pid>): Flow: ..."
format with bshell calls, depth-tracked function flow, and 3gl call returns).

100% LOCAL: this script only ever talks to Ollama at http://localhost:11434
(the default for the `ollama` Python client). Your trace data never leaves
your machine. No cloud/remote calls are made anywhere in this script.

WHY IT ASKS WHAT YOU WANT TO KNOW FIRST
-----------------------------------------
Different questions need different extraction strategies:
  - "why did this fail / what broke" -> scan for negative 3gl return codes
    (a genuine error signal in this trace format -- see note below)
  - "what happened with <order/function/customer X>" -> search the trace
    directly for that term and pull matching context, regardless of
    whether anything technically "failed"
Picking the wrong strategy either misses what you actually care about, or
wastes tokens dumping irrelevant matches. So the script asks first.

WHY NEGATIVE RETURN CODES (for error-type questions)
-------------------------------------------------------
Naive keyword scanning for "error", "NOT OK", "failed", "abort" is
UNRELIABLE here -- confirmed by direct comparison of matched success/failure
trace pairs: those strings show up at nearly identical rates in completely
successful traces too (DLL object-header noise, routine cache-miss retries).
Negative return codes on "Flow:3gl call returned: ( -N )" lines are a much
cleaner signal -- but NOTE: at large scale they are NOT automatically rare
(a 2M-line successful trace had 90 unique negative-return incidents of its
own, routine ones). The real signal is which incidents are unique to the
FAILURE trace when a baseline is available -- always provide one if you can.

TOKEN EFFICIENCY
-----------------
Both extraction strategies use plain regex/string search first (no LLM,
free, instant) to shrink a 50k-2M line file down to a handful of candidate
incidents. Only those handfuls get sent to the local LLM for explanation --
so a 2 million line trace might cost 10-20 LLM calls, not 2 million.

REQUIREMENTS
------------
pip install ollama psutil --break-system-packages
ollama pull llama3.2:3b     # or whatever local analysis model you have pulled

USAGE
-----
python3 infor_ln_analyzer.py <trace_file.gz> [baseline_trace.gz]
    (baseline is optional but recommended when you have it -- it lets the
    script tell routine noise apart from things unique to the failure)

You'll be prompted: "What do you want to know about this trace?"
Examples of things you can type:
    "why did this fail"
    "what caused the order print to break"
    "check for errors"
    "what happened with function create.sales.order.lines"
    "anything related to customer S21786802"

Output: prints a summary to console and writes a full JSON report to
./infor_ln_report.json
"""

import sys
import re
import gzip
import json
from pathlib import Path
from datetime import datetime, timezone

try:
    import ollama
except ImportError:
    print("Missing dependency. Run: pip install ollama --break-system-packages")
    sys.exit(1)

try:
    import psutil
except ImportError:
    psutil = None

# ============================== CONFIG ======================================

ANALYSIS_MODEL = "llama3.2:3b"     # local model used for explaining each incident
ANALYSIS_MAX_TOKENS = 220
CONTEXT_LINES_BEFORE = 4           # how many lines of context to keep before each incident
MAX_LINE_GAP_FOR_EVENT = 5         # occurrences within this many lines are treated as one event
REPORT_PATH = "./infor_ln_report.json"

MODEL_RAM_GB = {
    "qwen2.5:0.5b": 1.0,
    "qwen2.5:1.5b": 2.0,
    "llama3.2:1b":  1.5,
    "llama3.2:3b":  4.0,
    "phi3:mini":    2.5,
    "llama3.1:8b":  8.0,
}
RAM_SAFETY_BUFFER_GB = 1.0

NEG_RETURN_RE = re.compile(r"Flow:3gl call returned:\s*\(\s*(-\d+)\s*\)")
FUNC_NAME_RE = re.compile(r"depth\s+\d+\):\s*([\w.$]+)\s*\(")

STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "did", "does", "what", "why",
    "how", "when", "where", "this", "that", "trace", "check", "for",
    "with", "about", "happened", "caused", "cause", "broke", "broken",
    "fail", "failed", "failure", "error", "errors", "issue", "issues",
    "problem", "problems", "wrong", "not", "working", "anything",
    "related", "to", "and", "or", "in", "of", "on", "find", "show",
    "me", "please", "did", "there", "any",
}

ERROR_INTENT_WORDS = {
    "fail", "failed", "failure", "broke", "broken", "error", "errors",
    "wrong", "issue", "issues", "problem", "problems", "crash", "crashed",
    "not working", "why",
}

# =============================================================================


def estimate_ram_needed_gb() -> float:
    return MODEL_RAM_GB.get(ANALYSIS_MODEL, 5.0) + RAM_SAFETY_BUFFER_GB


def check_ram() -> bool:
    needed_gb = estimate_ram_needed_gb()
    if psutil is None:
        print(f"[RAM CHECK] psutil not installed, skipping. Est. need: ~{needed_gb:.1f} GB.")
        return True
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"[RAM CHECK] {available_gb:.1f} GB free of {total_gb:.1f} GB total | "
          f"model needs ~{needed_gb:.1f} GB ({ANALYSIS_MODEL} + buffer)")
    if available_gb >= needed_gb:
        print("[RAM CHECK] OK.")
        return True
    print(f"\n\u26a0\ufe0f  Only {available_gb:.1f} GB free, model needs ~{needed_gb:.1f} GB. "
          f"Close some apps or use a smaller ANALYSIS_MODEL.")
    try:
        return input("Continue anyway? [y/N]: ").strip().lower() == "y"
    except EOFError:
        return True


def read_lines(path: str) -> list:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", errors="ignore") as f:
        return f.readlines()


def get_query() -> str:
    if len(sys.argv) > 3 and sys.argv[3] == "--query" and len(sys.argv) > 4:
        return sys.argv[4]
    try:
        q = input("\nWhat do you want to know about this trace? "
                   "(e.g. 'why did this fail', 'what happened with order S21786802'): ").strip()
        return q or "why did this fail"
    except EOFError:
        return "why did this fail"


def classify_query(query: str):
    q_lower = query.lower()
    words = re.findall(r"[a-z0-9_.]+", q_lower)
    keywords = [w for w in words if w not in STOPWORDS and len(w) >= 3]
    has_error_intent = any(w in ERROR_INTENT_WORDS for w in words) or \
        "not working" in q_lower or "what broke" in q_lower
    # Real Infor LN identifiers always contain a dot, digit, or underscore --
    # plain long English words shouldn't count as specific technical terms.
    specific_terms = [w for w in keywords if re.search(r"[\d._]", w)]

    if specific_terms and not has_error_intent:
        return "keyword_scan", specific_terms
    if specific_terms and has_error_intent:
        return "error_scan_scoped", specific_terms
    return "error_scan", keywords


def find_negative_return_events(lines):
    raw = []
    for i, line in enumerate(lines):
        m = NEG_RETURN_RE.search(line)
        if not m:
            continue
        return_code = int(m.group(1))
        start = max(0, i - CONTEXT_LINES_BEFORE)
        context = [l.rstrip() for l in lines[start:i + 1]]
        func_name = "unknown"
        for ctx_line in reversed(context[:-1]):
            fm = FUNC_NAME_RE.search(ctx_line)
            if fm:
                func_name = fm.group(1)
                break
        raw.append({
            "line_number": i + 1,
            "return_code": return_code,
            "function": func_name,
            "context": context,
        })
    return raw


def cluster_events(raw, max_gap=MAX_LINE_GAP_FOR_EVENT):
    if not raw:
        return []
    raw_sorted = sorted(raw, key=lambda x: x["line_number"])
    events = [[raw_sorted[0]]]
    for occ in raw_sorted[1:]:
        if occ["line_number"] - events[-1][-1]["line_number"] <= max_gap:
            events[-1].append(occ)
        else:
            events.append([occ])
    return events


def summarize_events(events):
    summaries = []
    for event in events:
        root = event[0]
        call_chain = [e["function"] for e in event] if len(event) > 1 else None
        summaries.append({
            "function": root["function"],
            "return_code": root["return_code"],
            "call_chain": call_chain,
            "lines": [e["line_number"] for e in event],
            "context": root["context"],
        })
    return summaries


def dedupe_events(summaries):
    seen = {}
    for s in summaries:
        key = (s["function"], s["return_code"])
        if key not in seen:
            seen[key] = {**s, "occurrences": 1, "all_lines": list(s["lines"])}
        else:
            seen[key]["occurrences"] += 1
            seen[key]["all_lines"].extend(s["lines"])
    return list(seen.values())


def get_error_incidents(lines):
    raw = find_negative_return_events(lines)
    events = cluster_events(raw)
    summaries = summarize_events(events)
    return dedupe_events(summaries)


def diff_against_baseline(incidents, baseline_incidents):
    baseline_counts = {(i["function"], i["return_code"]): i["occurrences"] for i in baseline_incidents}
    SPIKE_RATIO = 3        # failure count must be at least this many times the baseline count
    SPIKE_MIN_DELTA = 5    # and the absolute difference must be at least this large
    for inc in incidents:
        key = (inc["function"], inc["return_code"])
        baseline_count = baseline_counts.get(key, 0)
        inc["baseline_occurrences"] = baseline_count
        if baseline_count == 0:
            inc["confidence"] = "high (absent from baseline trace)"
        elif inc["occurrences"] >= baseline_count * SPIKE_RATIO and \
                (inc["occurrences"] - baseline_count) >= SPIKE_MIN_DELTA:
            inc["confidence"] = (f"high (occurs {inc['occurrences']}x here vs only "
                                  f"{baseline_count}x in baseline -- {inc['occurrences'] / baseline_count:.0f}x spike)")
        else:
            inc["confidence"] = (f"low (occurs {baseline_count}x in baseline too, "
                                  f"similar rate -- routine)")
    return incidents


def relevance_filter(incidents, keywords):
    if not keywords:
        return incidents
    kept = []
    for inc in incidents:
        blob = (inc["function"] + " " + " ".join(inc["context"])).lower()
        if any(kw in blob for kw in keywords):
            kept.append(inc)
    return kept or incidents


def find_keyword_events(lines, keywords):
    pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
    raw = []
    for i, line in enumerate(lines):
        if pattern.search(line):
            start = max(0, i - CONTEXT_LINES_BEFORE)
            context = [l.rstrip() for l in lines[start:i + 1]]
            func_name = "unknown"
            for ctx_line in reversed(context):
                fm = FUNC_NAME_RE.search(ctx_line)
                if fm:
                    func_name = fm.group(1)
                    break
            raw.append({"line_number": i + 1, "function": func_name, "context": context,
                        "return_code": None})
    return raw


def get_keyword_incidents(lines, keywords, cap=15):
    raw = find_keyword_events(lines, keywords)
    if not raw:
        return []
    events = cluster_events(raw, max_gap=10)
    summaries = summarize_events(events)
    incidents = dedupe_events(summaries)
    incidents.sort(key=lambda x: -x["occurrences"])
    return incidents[:cap]


ERROR_ANALYSIS_SYSTEM = (
    "You are a senior Infor LN / Baan ERP support engineer. You'll be shown a "
    "snippet from a 4GL runtime trace where a function call ('3gl call') "
    "returned a negative value, which indicates an error status in this system. "
    "The user asked a specific question about this trace -- keep your answer "
    "focused on what they asked. Explain in plain English: (1) what business "
    "function was being called (infer from the function name and string "
    "arguments), (2) what a negative return code like this typically signals "
    "here, (3) a concrete next step to investigate. Reply with concise JSON "
    "only, no markdown fences: "
    '{"summary":"...", "likely_cause":"...", "next_step":"..."}. Keep each '
    "field under 30 words."
)

KEYWORD_ANALYSIS_SYSTEM = (
    "You are a senior Infor LN / Baan ERP support engineer. The user asked a "
    "specific question about a runtime trace. You'll be shown a snippet where "
    "their search term(s) appeared. Explain in plain English, focused ONLY on "
    "answering their question: (1) what is happening in this snippet relevant "
    "to their question, (2) whether anything here looks like a problem or is "
    "just normal flow, (3) a concrete next step if relevant. Reply with concise "
    'JSON only, no markdown fences: {"summary":"...", "likely_cause":"...", '
    '"next_step":"..."}. Keep each field under 30 words. If this snippet looks '
    "like completely normal/successful operation, say so plainly instead of "
    "inventing a problem."
)


def explain_incident(incident, query, mode):
    snippet = "\n".join(incident["context"])
    chain_note = ""
    if incident.get("call_chain") and len(incident["call_chain"]) > 1:
        chain_note = (
            f"\nThis propagated through {len(incident['call_chain'])} call-stack "
            f"frames: {' -> '.join(incident['call_chain'])} (innermost first).\n"
        )
    code_line = f"Return code: {incident['return_code']}\n" if incident.get("return_code") is not None else ""
    prompt = (
        f"User's question: \"{query}\"\n\n"
        f"Function: {incident['function']}\n"
        f"{code_line}"
        f"Occurred {incident['occurrences']} time(s), at line(s) {incident['all_lines'][:10]}"
        f"{chain_note}\n"
        f"Trace context:\n{snippet}"
    )
    system = ERROR_ANALYSIS_SYSTEM if mode.startswith("error_scan") else KEYWORD_ANALYSIS_SYSTEM
    resp = ollama.chat(
        model=ANALYSIS_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={"num_predict": ANALYSIS_MAX_TOKENS, "temperature": 0},
    )
    raw = resp["message"]["content"].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"summary": raw[:300], "likely_cause": "", "next_step": ""}


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 infor_ln_analyzer.py <trace_file.gz> [baseline_trace.gz] [--query \"...\"]")
        sys.exit(1)

    failure_path = sys.argv[1]
    baseline_path = None
    if len(sys.argv) > 2 and sys.argv[2] != "--query":
        baseline_path = sys.argv[2]

    if not check_ram():
        print("Aborted.")
        sys.exit(0)

    query = get_query()
    mode, keywords = classify_query(query)
    print(f"\n[QUERY] \"{query}\"")
    print(f"[MODE] {mode}" + (f" | keywords: {keywords}" if keywords else ""))

    print(f"\nReading trace: {failure_path}")
    failure_lines = read_lines(failure_path)
    print(f"  {len(failure_lines):,} lines")

    if mode == "keyword_scan":
        incidents = get_keyword_incidents(failure_lines, keywords)
        for inc in incidents:
            inc["confidence"] = "n/a (keyword match, not error-based)"
        baseline_stats = None
        print(f"Found {len(incidents)} matching event(s) for: {keywords}")

    else:
        incidents = get_error_incidents(failure_lines)
        print(f"Found {len(incidents)} unique failure event(s) (after event clustering + dedup).")

        baseline_stats = None
        if baseline_path:
            print(f"\nReading baseline trace: {baseline_path}")
            baseline_lines = read_lines(baseline_path)
            baseline_incidents = get_error_incidents(baseline_lines)
            print(f"  Baseline has {len(baseline_incidents)} unique event(s) of its own.")
            incidents = diff_against_baseline(incidents, baseline_incidents)
            baseline_stats = {
                "file": baseline_path,
                "total_lines_scanned": len(baseline_lines),
                "unique_incidents_in_baseline": len(baseline_incidents),
            }
        else:
            for inc in incidents:
                inc["confidence"] = "unverified (no baseline trace provided)"

        if mode == "error_scan_scoped":
            before = len(incidents)
            incidents = relevance_filter(incidents, keywords)
            print(f"Filtered to {len(incidents)}/{before} event(s) relevant to: {keywords}")

    if not incidents:
        print("\nNothing found matching your question in this trace.")
        Path(REPORT_PATH).write_text(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_file": failure_path, "query": query, "mode": mode, "incidents": [],
        }, indent=2))
        return

    if baseline_path and mode.startswith("error_scan"):
        to_analyze = [i for i in incidents if i["confidence"].startswith("high")]
        skipped = [i for i in incidents if not i["confidence"].startswith("high")]
        print(f"\n{len(to_analyze)} event(s) unique to the failure trace -- analyzing those.")
        print(f"{len(skipped)} event(s) also occur in the baseline -- skipping LLM calls "
              f"for those (routine, not worth explaining).\n")
    else:
        to_analyze = incidents
        skipped = []

    print(f"Analyzing {len(to_analyze)} event(s) with {ANALYSIS_MODEL}...\n")
    report = []

    for inc in skipped:
        report.append({
            "function": inc["function"], "return_code": inc.get("return_code"),
            "occurrences": inc["occurrences"], "line_numbers": inc["all_lines"],
            "call_chain": inc.get("call_chain"), "confidence": inc["confidence"],
            "explanation": "skipped -- also occurs in baseline trace, likely routine",
            "raw_context": inc["context"],
        })

    for inc in to_analyze:
        explanation = explain_incident(inc, query, mode)
        record = {
            "function": inc["function"], "return_code": inc.get("return_code"),
            "occurrences": inc["occurrences"], "line_numbers": inc["all_lines"],
            "call_chain": inc.get("call_chain"), "confidence": inc["confidence"],
            "explanation": explanation, "raw_context": inc["context"],
        }
        report.append(record)

        chain_str = f" | chain: {' -> '.join(inc['call_chain'])}" if inc.get("call_chain") else ""
        code_str = f"returned {inc['return_code']} " if inc.get("return_code") is not None else ""
        print(f"--- {inc['function']} {code_str}"
              f"({inc['occurrences']}x, confidence: {inc['confidence']}){chain_str} ---")
        print(f"  Summary:     {explanation.get('summary', '')}")
        print(f"  Likely cause:{explanation.get('likely_cause', '')}")
        print(f"  Next step:   {explanation.get('next_step', '')}\n")

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_file": failure_path,
        "query": query,
        "mode": mode,
        "baseline_comparison": baseline_stats,
        "total_lines_scanned": len(failure_lines),
        "incidents": report,
    }
    Path(REPORT_PATH).write_text(json.dumps(out, indent=2))
    print(f"Full report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
