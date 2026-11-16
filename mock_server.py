#!/usr/bin/env python3
"""Local stand-in for the Anthropic Messages, OpenAI Chat Completions and
Adaption Responses (POST /api/v1/responses, GET /api/v1/models) APIs.

FOR TESTING THE PIPELINE ONLY. Every response is SYNTHETIC: the probability is
computed from the prompt with fixed, known effects so the self-test can check
that analyze.py recovers them. Nothing it returns is, or resembles, a result
from a real model. The model id it reports always contains "mock".

    python mock_server.py [--port 0] [--log FILE] [--slow-after N]

Prints "PORT <n>" on its first stdout line. Binds to 127.0.0.1 only.

Injected effects (probability points): unlabelled +12, attributed +4,
evidence +25, added to a per-case base of 20-40. Repeats cycle -2, 0, +2.

Fault schedule, by request number (counting every HTTP request, retries too):
  3  HTTP 500           (SDK should retry; the retry succeeds)
  8  HTTP 400           (not retried; recorded as api_error)
  12 prose, no JSON     -> no_json
  16 cut-off JSON       -> truncated (stop reason = token limit)
  20 probability 150    -> out_of_range
  24 refusal            -> refusal
  28 JSON in a fence    -> still parses (flag: code_fence)
  32 probability 0.35   -> ambiguous_scale
  36 Adaption only: HTTP 200 with status "failed" -> response_failed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MOCK_MODEL = "mock-model-NOT-A-REAL-MODEL"
EFFECT_UNLABELLED, EFFECT_ATTRIBUTED, EFFECT_EVIDENCE, EFFECT_DISCONFIRMING = 12, 4, 25, -15
try:  # the mock recognises disconfirming evidence by its text in the v3 cases file
    from pathlib import Path as _P
    _V3 = json.loads((_P(__file__).resolve().parent / "pilot_cases_v3.json").read_text(encoding="utf-8"))
    DISCONFIRMING_TEXTS = [c["disconfirming_evidence"] for c in _V3["cases"] if c.get("disconfirming_evidence")]
except (OSError, ValueError, KeyError):
    DISCONFIRMING_TEXTS = []
REPEAT_OFFSETS = (-2, 0, 2)
FAULTS = {3: "http500", 8: "http400", 12: "prose", 16: "truncated", 20: "out_of_range",
          24: "refusal", 28: "fenced", 32: "fraction", 36: "failed_status"}
MOCK_MODEL_LIST = [
    {"id": MOCK_MODEL, "created": 1750000000, "object": "model", "owned_by": "mock",
     "display_name": "SYNTHETIC mock model", "shutdown_date": None},
    {"id": "mock-retiring-model", "created": 1750000000, "object": "model", "owned_by": "mock",
     "display_name": "SYNTHETIC retiring model", "shutdown_date": "2027-01-01"},
]

FILLER_A = "It was suggested that the conversation continue another time."
FILLER_B = 'The user later added: "Thanks for your help so far."'


def classify(system: str) -> tuple[str, str, str]:
    lines = system.splitlines()
    obs = next((l for l in lines if l.startswith("- The user reported:")), "")
    if "- The assistant suggested that " in system:
        a = "attributed"
    elif FILLER_A in system:
        a = "filler"
    elif "- It was suggested that " in system:
        a = "unlabelled"
    else:
        a = "unknown"
    if FILLER_B in system:
        b = "filler"
    elif any(t in system for t in DISCONFIRMING_TEXTS):
        b = "disconfirming"
    elif "- The user later added:" in system:
        b = "evidence"
    else:
        b = "unknown"
    return obs, a, b


def effect_b(b: str) -> int:
    return {"evidence": EFFECT_EVIDENCE, "disconfirming": EFFECT_DISCONFIRMING}.get(b, 0)


class State:
    def __init__(self, log_path, slow_after, slow_seconds):
        self.lock = threading.Lock()
        self.n = 0
        self.per_cell = Counter()
        self.log_path = log_path
        self.slow_after = slow_after
        self.slow_seconds = slow_seconds


class Handler(BaseHTTPRequestHandler):
    state: State = None  # set in main

    def log_message(self, *a):  # keep stdout clean
        pass

    def _send(self, code, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path != "/api/v1/models":
            return self._send(404, {"error": {"message": f"unknown path {self.path}"}})
        if not self.headers.get("Authorization"):
            return self._send(401, {"error": {"message": "missing credentials"}})
        return self._send(200, {"object": "list", "data": MOCK_MODEL_LIST}, {"x-request-id": "req_mock_models"})

    def do_POST(self):
        st = self.state
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.loads(raw or b"{}")
        is_anthropic = self.path.rstrip("/").endswith("/v1/messages")
        is_openai = self.path.rstrip("/").endswith("/chat/completions")
        is_adaption = self.path.rstrip("/").endswith("/api/v1/responses")
        if is_adaption:
            return self._adaption(body)
        with st.lock:
            st.n += 1
            n = st.n
        fault = FAULTS.get(n)
        if fault == "failed_status":
            fault = None  # Adaption-only fault
        if st.log_path:
            with st.lock, open(st.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"n": n, "path": self.path, "fault": fault, "body": body}) + "\n")
        if st.slow_after is not None and n > st.slow_after:
            time.sleep(st.slow_seconds)

        # anthropic SDK 2.x sends the key as Authorization; older versions used x-api-key.
        auth_ok = bool(self.headers.get("Authorization") or (is_anthropic and self.headers.get("x-api-key")))
        rid = {"request-id" if is_anthropic else "x-request-id": f"req_mock_{n}"}
        if not (is_anthropic or is_openai):
            return self._send(404, {"error": {"message": f"unknown path {self.path}"}})
        if not auth_ok:
            return self._send(401, {"error": {"message": "missing credentials"}}, rid)

        if fault == "http500":
            err = {"type": "error", "error": {"type": "api_error", "message": "mock internal error"}} if is_anthropic \
                else {"error": {"message": "mock internal error", "type": "server_error", "code": None}}
            return self._send(500, err, rid)
        if fault == "http400":
            err = {"type": "error", "error": {"type": "invalid_request_error", "message": "mock bad request"}} if is_anthropic \
                else {"error": {"message": "mock bad request", "type": "invalid_request_error", "code": None}}
            return self._send(400, err, rid)

        system = body.get("system", "") if is_anthropic else next(
            (m["content"] for m in body.get("messages", []) if m.get("role") == "system"), "")
        obs, a, b = classify(system)
        base = 20 + int(hashlib.sha256(obs.encode()).hexdigest()[:8], 16) % 21
        p = base + {"unlabelled": EFFECT_UNLABELLED, "attributed": EFFECT_ATTRIBUTED}.get(a, 0) \
            + effect_b(b)

        stop = "end_turn" if is_anthropic else "stop"
        if fault is None:
            with st.lock:
                k = st.per_cell[(obs, a, b)]
                st.per_cell[(obs, a, b)] += 1
            p = max(0, min(100, p + REPEAT_OFFSETS[k % len(REPEAT_OFFSETS)]))
        reason = "SYNTHETIC mock response; not a model output."
        text = json.dumps({"probability": p, "reason": reason})
        if fault == "prose":
            text = "SYNTHETIC: I'd say it's fairly likely, maybe around 60%."
        elif fault == "truncated":
            text, stop = '{"probability": 4', ("max_tokens" if is_anthropic else "length")
        elif fault == "out_of_range":
            text = json.dumps({"probability": 150, "reason": reason})
        elif fault == "fenced":
            text = "```json\n" + text + "\n```"
        elif fault == "fraction":
            text = json.dumps({"probability": 0.35, "reason": reason})
        refusal = fault == "refusal"

        if is_anthropic:
            resp = {
                "id": f"msg_mock_{n}", "type": "message", "role": "assistant", "model": MOCK_MODEL,
                "content": [] if refusal else [{"type": "text", "text": text}],
                "stop_reason": "refusal" if refusal else stop, "stop_sequence": None,
                "usage": {"input_tokens": len(system.split()), "output_tokens": len(text.split())},
            }
        else:
            resp = {
                "id": f"chatcmpl-mock-{n}", "object": "chat.completion", "created": int(time.time()), "model": MOCK_MODEL,
                "choices": [{"index": 0, "finish_reason": "content_filter" if refusal else stop,
                             "message": {"role": "assistant", "content": None if refusal else text}}],
                "usage": {"prompt_tokens": len(system.split()), "completion_tokens": len(text.split()),
                          "total_tokens": len(system.split()) + len(text.split())},
            }
        return self._send(200, resp, rid)

    def _adaption(self, body):
        """Responses-API shape, following docs.adaptionlabs.ai/api/resources/responses/methods/create."""
        st = self.state
        with st.lock:
            st.n += 1
            n = st.n
        fault = FAULTS.get(n)
        if st.log_path:
            with st.lock, open(st.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"n": n, "path": self.path, "fault": fault, "body": body}) + "\n")
        if st.slow_after is not None and n > st.slow_after:
            time.sleep(st.slow_seconds)
        rid = {"x-request-id": f"req_mock_{n}"}
        if not self.headers.get("Authorization"):
            return self._send(401, {"error": {"code": "unauthorized", "message": "missing credentials"}}, rid)
        if fault == "http500":
            return self._send(500, {"error": {"code": "internal_error", "message": "mock internal error"}}, rid)
        if fault == "http400":
            return self._send(400, {"error": {"code": "invalid_request", "message": "mock bad request"}}, rid)

        system = body.get("instructions") or ""
        obs, a, b = classify(system)
        base = 20 + int(hashlib.sha256(obs.encode()).hexdigest()[:8], 16) % 21
        p = base + {"unlabelled": EFFECT_UNLABELLED, "attributed": EFFECT_ATTRIBUTED}.get(a, 0) \
            + effect_b(b)
        if fault is None:
            with st.lock:
                k = st.per_cell[(obs, a, b)]
                st.per_cell[(obs, a, b)] += 1
            p = max(0, min(100, p + REPEAT_OFFSETS[k % len(REPEAT_OFFSETS)]))
        reason = "SYNTHETIC mock response; not a model output."
        text = json.dumps({"probability": p, "reason": reason})
        status, incomplete, error = "completed", None, None
        if fault == "prose":
            text = "SYNTHETIC: I'd say it's fairly likely, maybe around 60%."
        elif fault == "truncated":
            text, status, incomplete = '{"probability": 4', "incomplete", {"reason": "max_output_tokens"}
        elif fault == "out_of_range":
            text = json.dumps({"probability": 150, "reason": reason})
        elif fault == "fenced":
            text = "```json\n" + text + "\n```"
        elif fault == "fraction":
            text = json.dumps({"probability": 0.35, "reason": reason})
        elif fault == "failed_status":
            text, status, error = "", "failed", {"code": "server_error", "message": "SYNTHETIC generation failure"}
        content = [{"type": "refusal", "refusal": "SYNTHETIC refusal."}] if fault == "refusal" \
            else ([{"type": "output_text", "text": text, "annotations": []}] if text else [])
        n_in, n_out = len(system.split()) + len(str(body.get("input")).split()), len(text.split())
        resp = {
            "id": f"resp_mock_{n}", "object": "response", "created_at": int(time.time()), "model": MOCK_MODEL,
            "status": status,
            "output": [{"type": "message", "id": f"msg_mock_{n}", "status": status, "role": "assistant", "content": content}],
            "output_text": "" if fault == "refusal" else text,
            "error": error, "incomplete_details": incomplete,
            "usage": {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
            "store": body.get("store"),
        }
        return self._send(200, resp, rid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--log", default=None, help="append every received request body (no headers) to this JSONL file")
    ap.add_argument("--slow-after", type=int, default=None, help="delay every request after this number")
    ap.add_argument("--slow-seconds", type=float, default=60.0)
    a = ap.parse_args()
    Handler.state = State(a.log, a.slow_after, a.slow_seconds)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    print(f"PORT {srv.server_address[1]}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
