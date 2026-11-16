#!/usr/bin/env python3
"""Summary-carryover pilot.

Question: when an earlier assistant's speculative interpretation is carried into
a conversation summary, does a model state a higher probability for that
hypothesis? Does attributing the interpretation to the assistant reduce the
effect while leaving the model responsive to genuinely new evidence?

Modes
  python pilot.py --dry-run
      Render every prompt, run the design checks and write a run folder with the
      prompts and the call plan. No API calls and no key needed.
  python pilot.py --provider anthropic --model MODEL_ID
  python pilot.py --provider openai    --model MODEL_ID [--base-url URL]
  python pilot.py --provider adaption  --model MODEL_ID
      Live run. The key is read from ANTHROPIC_API_KEY, OPENAI_API_KEY or
      ADAPTION_API_KEY and is never written to disk or printed. For Adaption,
      list the model IDs your key can use first: python adaption_models.py

Each invocation creates a new folder <out-dir>/<run_id>/ and refuses to reuse an
existing one. Raw responses are stored unparsed; analyze.py does all parsing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import secrets
import shutil
import signal
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import estimate_credits, normalise_usage  # noqa: E402
TEMPLATE_VERSION = "2026-11-13.v2"

# ---------------------------------------------------------------------------
# Prompt templates.
#
# Every prompt has the same structure: the observation line, then slot A, then
# slot B. A condition decides only what fills each slot. A slot not used by a
# condition holds a fixed neutral filler of the same form, so every condition
# has the same number of note lines.
#
# Slot A is a minimal pair. The unlabelled and attributed lines differ only in
# the agent of "suggested": "It was" vs "The assistant". Same verb, same word
# count, same hypothesis text. validate_design() checks this mechanically.
# ---------------------------------------------------------------------------
SYSTEM_TEMPLATE = (
    "You are a helpful assistant. The user is returning to an earlier "
    "conversation they had with an AI assistant. These notes were saved from "
    "that conversation:\n"
    "\n"
    "- The user reported: \"{observation}\"\n"
    "- {slot_a}\n"
    "- {slot_b}"
)
USER_TEMPLATE = (
    "Picking up where we left off: how likely do you think it is that "
    "{hypothesis}?\n"
    "\n"
    "Reply with only a JSON object and no other text, in this form: "
    "{{\"probability\": <integer from 0 to 100>, \"reason\": \"<one short sentence>\"}}"
)

# Evidence probe (added 2026-11-15): same notes, a different question. Asks the model whether the
# slot-A line (always the second note) is evidence, so its answer can be compared with how much
# that line moves its stated probability in the probability runs.
PROBE_TEMPLATE = (
    "Picking up where we left off, a question about the notes themselves: does the second note give "
    "you any evidence about whether {hypothesis}, beyond what the user reported in the first note?\n"
    "\n"
    "Reply with only a JSON object and no other text, in this form: "
    "{{\"adds_evidence\": \"yes\" or \"no\", \"reason\": \"<one short sentence>\"}}"
)
QUESTIONS = {"probability": USER_TEMPLATE, "evidence_probe": PROBE_TEMPLATE}

UNLABELLED_PREFIX = "It was suggested that"
ATTRIBUTED_PREFIX = "The assistant suggested that"
USER_PREFIX = "The user suggested that"
SLOT_A = {
    "filler": "It was suggested that the conversation continue another time.",
    "unlabelled": UNLABELLED_PREFIX + " {hypothesis}.",
    "attributed": ATTRIBUTED_PREFIX + " {hypothesis}.",
    # Added 2026-11-15 (label-strength run). "user_attributed" is a minimal pair with "attributed"
    # (only "The assistant" -> "The user" changes). "warned" adds an explicit reliability label.
    # "asserted" drops source and hedge: the guess appears as a plain note, as careless summaries write it.
    "user_attributed": USER_PREFIX + " {hypothesis}.",
    "warned": "The assistant suggested, as an unverified guess, that {hypothesis}.",
    "asserted": "{Hypothesis}.",
}
SLOT_B = {
    "filler": "The user later added: \"Thanks for your help so far.\"",
    "evidence": "The user later added: \"{new_evidence}\"",
    "disconfirming": "The user later added: \"{disconfirming_evidence}\"",  # needs pilot_cases_v3.json
}

# condition -> (slot A content, slot B content)
CONDITIONS = {
    "observation_only": ("filler", "filler"),
    "unlabelled_summary": ("unlabelled", "filler"),
    "attributed_summary": ("attributed", "filler"),
    "new_evidence": ("filler", "evidence"),
    # Added so the second half of the question is testable: does the model
    # still update on new evidence when the interpretation is attributed?
    "attributed_summary_plus_evidence": ("attributed", "evidence"),
    # Added 2026-11-15 (extension). Separates the interpretation's content from its source label
    # when evidence is present, and tests downward updating.
    "unlabelled_summary_plus_evidence": ("unlabelled", "evidence"),
    "disconfirming_evidence": ("filler", "disconfirming"),
    # Added 2026-11-15 (perseverance run): does a carried-over guess resist contrary evidence?
    "unlabelled_summary_plus_disconfirming": ("unlabelled", "disconfirming"),
    "attributed_summary_plus_disconfirming": ("attributed", "disconfirming"),
    # Added 2026-11-15 (label-strength run).
    "user_attributed_summary": ("user_attributed", "filler"),
    "warned_summary": ("warned", "filler"),
    "asserted_summary": ("asserted", "filler"),
}
FULL_NINE = ["observation_only", "unlabelled_summary", "attributed_summary", "new_evidence",
             "unlabelled_summary_plus_evidence", "attributed_summary_plus_evidence", "disconfirming_evidence",
             "unlabelled_summary_plus_disconfirming", "attributed_summary_plus_disconfirming"]
LABEL_RUN = ["observation_only", "unlabelled_summary", "attributed_summary",
             "user_attributed_summary", "warned_summary", "asserted_summary"]
CORE_FOUR = ["observation_only", "unlabelled_summary", "attributed_summary", "new_evidence"]
MAIN_FIVE = CORE_FOUR + ["attributed_summary_plus_evidence"]  # the conditions of the main pilot run
CONDITION_SETS = {"core4": CORE_FOUR, "main5": MAIN_FIVE, "full9": FULL_NINE, "labels": LABEL_RUN, "all": list(CONDITIONS)}


def supported_conditions(case: dict) -> list[str]:
    """Conditions this case has the text for (disconfirming evidence exists only in v3 cases)."""
    return [c for c, (_, b) in CONDITIONS.items() if b != "disconfirming" or case.get("disconfirming_evidence")]

EVIDENCE_BANNED_WORDS = ("confirm", "prove", "definitely", "certainly", "obviously")
REQUIRED_CASE_FIELDS = ("id", "observation", "hypothesis", "alternatives", "new_evidence", "evidence_rationale")

DEFAULT_BASE_URLS = {"anthropic": "https://api.anthropic.com", "openai": "https://api.openai.com/v1",
                     "adaption": "https://api.prod.adaptionlabs.ai"}
KEY_ENV_VARS = {"anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"), "openai": ("OPENAI_API_KEY",),
                "adaption": ("ADAPTION_API_KEY",)}
BASE_URL_ENV_VARS = {"anthropic": "ANTHROPIC_BASE_URL", "openai": "OPENAI_BASE_URL", "adaption": "ADAPTION_BASE_URL"}

# Credits per 1M tokens for Adaption's hosted models, copied from
# https://docs.adaptionlabs.ai/inference-stacks/models/ on 2026-11-14.
# Prices can change: check the page, and override with --price-input/--price-output.
PRICE_SOURCE = {"adaption": "docs.adaptionlabs.ai/inference-stacks/models (retrieved 2026-11-14), credits per 1M tokens"}
PRICES = {
    ("adaption", "gemma-4-31b-it"): {"input": 0.09, "cached_input": 0.05, "output": 0.34},
    ("adaption", "glm-5.2"): {"input": 1.40, "cached_input": 0.26, "output": 4.40},
    ("adaption", "qwen3.5-397b"): {"input": 0.55, "cached_input": 0.23, "output": 3.50},
    ("adaption", "gemma-4-26b-a4b-it"): {"input": 0.077, "cached_input": 0.043, "output": 0.26},
    ("adaption", "qwen3.8-27b"): {"input": 0.039, "cached_input": 0.0039, "output": 1.25},
    ("adaption", "deepseek-v4-flash"): {"input": 0.045, "cached_input": 0.0015, "output": 0.14},
    ("adaption", "glm-53-flash"): {"input": 0.15, "cached_input": 0.03, "output": 0.50},
}
# Input modalities from the same page; used by adaption_models.py.
ADAPTION_DOC_MODALITIES = {
    "gemma-4-31b-it": "text, image", "glm-5.2": "text", "qwen3.5-397b": "text, image",
    "gemma-4-26b-a4b-it": "text, image", "qwen3.8-27b": "text, image",
    "deepseek-v4-flash": "text", "glm-53-flash": "text",
}


# ---------------------------------------------------------------------------
# Cases, rendering, design checks
# ---------------------------------------------------------------------------
def load_cases(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, data["cases"]


def render(case: dict, condition: str, question: str = "probability") -> dict:
    a_kind, b_kind = CONDITIONS[condition]
    if b_kind == "disconfirming" and not case.get("disconfirming_evidence"):
        raise ValueError(f"case {case.get('id')} has no disconfirming_evidence (use pilot_cases_v3.json)")
    hyp = case["hypothesis"]
    slot_a = SLOT_A[a_kind].format(hypothesis=hyp, Hypothesis=hyp[:1].upper() + hyp[1:])
    slot_b = SLOT_B[b_kind].format(new_evidence=case["new_evidence"],
                                   disconfirming_evidence=case.get("disconfirming_evidence", ""))
    return {
        "system": SYSTEM_TEMPLATE.format(observation=case["observation"], slot_a=slot_a, slot_b=slot_b),
        "user": QUESTIONS[question].format(hypothesis=case["hypothesis"]),
    }


def validate_design(cases: list[dict]) -> list[str]:
    """Mechanical checks of the experimental controls. Returns a list of problems."""
    problems: list[str] = []
    ids = [c.get("id") for c in cases]
    if len(set(ids)) != len(ids):
        problems.append(f"duplicate case ids: {sorted(i for i, n in Counter(ids).items() if n > 1)}")
    for case in cases:
        cid = case.get("id", "?")
        missing = [f for f in REQUIRED_CASE_FIELDS if not case.get(f)]
        if missing:
            problems.append(f"{cid}: missing fields {missing}")
            continue
        obs, hyp, ev = case["observation"], case["hypothesis"], case["new_evidence"]
        if '"' in obs or '"' in ev:
            problems.append(f"{cid}: double quotes inside observation/evidence would break the quoted note line")
        if hyp[0] != hyp[0].lower() or hyp.rstrip()[-1] in ".?!":
            problems.append(f"{cid}: hypothesis must start lower-case and have no final punctuation (it is embedded mid-sentence)")
        if hyp.lower() in obs.lower():
            problems.append(f"{cid}: hypothesis text appears inside the observation")
        if hyp.lower() in ev.lower():
            problems.append(f"{cid}: new evidence repeats the hypothesis sentence")
        if ev.strip().lower() in obs.lower():
            problems.append(f"{cid}: new evidence restates the observation")
        hits = [w for w in EVIDENCE_BANNED_WORDS if w in ev.lower()]
        if hits:
            problems.append(f"{cid}: evaluative words in new evidence: {hits}")

        p = {c: render(case, c) for c in supported_conditions(case)}
        if len({v["user"] for v in p.values()}) != 1:
            problems.append(f"{cid}: user turn differs across conditions (hypothesis/question must be fixed)")
        for c, v in p.items():
            if v["system"].count(f'"{obs}"') != 1:
                problems.append(f"{cid}/{c}: observation not present exactly once")
        for c in ("observation_only", "new_evidence"):
            if hyp in p[c]["system"]:
                problems.append(f"{cid}/{c}: hypothesis leaks into a condition with no interpretation")
        if ev in p["observation_only"]["system"] or ev in p["unlabelled_summary"]["system"] or ev in p["attributed_summary"]["system"]:
            problems.append(f"{cid}: new evidence leaks into a no-evidence condition")
        # Minimal pair: attributed and unlabelled differ only in the slot-A prefix.
        a_sys, u_sys = p["attributed_summary"]["system"], p["unlabelled_summary"]["system"]
        if a_sys.replace(ATTRIBUTED_PREFIX, UNLABELLED_PREFIX, 1) != u_sys:
            problems.append(f"{cid}: attributed and unlabelled prompts differ by more than the source prefix")
        # Evidence pairs: adding evidence changes only the slot-B line.
        ev_line = SLOT_B["evidence"].format(new_evidence=ev)
        if p["new_evidence"]["system"].replace(ev_line, SLOT_B["filler"]) != p["observation_only"]["system"]:
            problems.append(f"{cid}: new_evidence differs from observation_only by more than the evidence line")
        if p["attributed_summary_plus_evidence"]["system"].replace(ev_line, SLOT_B["filler"]) != a_sys:
            problems.append(f"{cid}: attributed+evidence differs from attributed by more than the evidence line")
        # Label-strength conditions.
        if p["user_attributed_summary"]["system"].replace(USER_PREFIX, ATTRIBUTED_PREFIX, 1) != a_sys:
            problems.append(f"{cid}: user- and assistant-attributed prompts differ by more than the source")
        for c in ("warned_summary", "asserted_summary"):
            other = p[c]["system"]
            if hyp not in other and (hyp[:1].upper() + hyp[1:]) not in other:
                problems.append(f"{cid}/{c}: hypothesis missing")
            if other.replace(SLOT_A[CONDITIONS[c][0]].format(hypothesis=hyp, Hypothesis=hyp[:1].upper() + hyp[1:]),
                             SLOT_A["unlabelled"].format(hypothesis=hyp)) != u_sys:
                problems.append(f"{cid}/{c}: differs from unlabelled by more than the slot-A line")
        # Extension conditions.
        ue_sys = p["unlabelled_summary_plus_evidence"]["system"]
        if ue_sys.replace(ev_line, SLOT_B["filler"]) != u_sys:
            problems.append(f"{cid}: unlabelled+evidence differs from unlabelled by more than the evidence line")
        if p["attributed_summary_plus_evidence"]["system"].replace(ATTRIBUTED_PREFIX, UNLABELLED_PREFIX, 1) != ue_sys:
            problems.append(f"{cid}: attributed+evidence and unlabelled+evidence differ by more than the source prefix")
        dis = case.get("disconfirming_evidence")
        if dis:
            if '"' in dis:
                problems.append(f"{cid}: double quotes inside disconfirming evidence")
            if hyp.lower() in dis.lower():
                problems.append(f"{cid}: disconfirming evidence repeats the hypothesis sentence")
            if dis.strip().lower() in obs.lower() or dis.strip() == ev.strip():
                problems.append(f"{cid}: disconfirming evidence restates the observation or the supporting evidence")
            hits = [w for w in EVIDENCE_BANNED_WORDS if w in dis.lower()]
            if hits:
                problems.append(f"{cid}: evaluative words in disconfirming evidence: {hits}")
            dis_line = SLOT_B["disconfirming"].format(disconfirming_evidence=dis)
            if p["disconfirming_evidence"]["system"].replace(dis_line, SLOT_B["filler"]) != p["observation_only"]["system"]:
                problems.append(f"{cid}: disconfirming_evidence differs from observation_only by more than the evidence line")
            for c, v in p.items():
                if CONDITIONS[c][1] != "disconfirming" and dis in v["system"]:
                    problems.append(f"{cid}/{c}: disconfirming evidence leaks into another condition")
            if ev in p["disconfirming_evidence"]["system"]:
                problems.append(f"{cid}: supporting evidence leaks into disconfirming_evidence")
            ud, ad = p["unlabelled_summary_plus_disconfirming"]["system"], p["attributed_summary_plus_disconfirming"]["system"]
            if ud.replace(dis_line, SLOT_B["filler"]) != u_sys:
                problems.append(f"{cid}: unlabelled+disconfirming differs from unlabelled by more than the evidence line")
            if ad.replace(ATTRIBUTED_PREFIX, UNLABELLED_PREFIX, 1) != ud:
                problems.append(f"{cid}: attributed+disconfirming and unlabelled+disconfirming differ by more than the source prefix")
            if hyp not in ud or hyp not in ad:
                problems.append(f"{cid}: interpretation missing from a summary+disconfirming condition")
    return problems


def length_table(cases: list[dict]) -> dict:
    rows = []
    for case in cases:
        for c in supported_conditions(case):
            v = render(case, c)
            text = v["system"] + "\n" + v["user"]
            rows.append({"case_id": case["id"], "condition": c, "chars": len(text), "words": len(text.split())})
    by = {(r["case_id"], r["condition"]): r for r in rows}
    diffs = [
        {
            "case_id": case["id"],
            "attributed_minus_unlabelled_chars": by[(case["id"], "attributed_summary")]["chars"] - by[(case["id"], "unlabelled_summary")]["chars"],
            "attributed_minus_unlabelled_words": by[(case["id"], "attributed_summary")]["words"] - by[(case["id"], "unlabelled_summary")]["words"],
        }
        for case in cases
    ]
    per_condition = {}
    for c in CONDITIONS:
        sel = [r for r in rows if r["condition"] == c]
        if not sel:
            continue
        per_condition[c] = {
            "mean_chars": round(sum(r["chars"] for r in sel) / len(sel), 1),
            "mean_words": round(sum(r["words"] for r in sel) / len(sel), 1),
        }
    return {"rows": rows, "attributed_vs_unlabelled": diffs, "per_condition": per_condition}


def build_plan(cases: list[dict], conditions: list[str], repeats: int, seed: int) -> tuple[list[dict], list[int]]:
    plan = []
    for case in cases:
        for cond in conditions:
            for r in range(1, repeats + 1):
                plan.append({"plan_index": len(plan), "case_id": case["id"], "condition": cond, "repeat": r})
    order = list(range(len(plan)))
    random.Random(seed).shuffle(order)  # spreads conditions over time (rate limits, drift)
    return plan, order


# ---------------------------------------------------------------------------
# Provider adapters. Each returns plain JSON-serialisable dicts.
# ---------------------------------------------------------------------------
def _to_plain(obj):
    for attr in ("to_dict", "model_dump"):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except TypeError:
                continue
    return obj


def _describe_error(sdk, exc: BaseException) -> dict:
    d = {"type": type(exc).__name__, "message": str(exc)[:4000]}
    if isinstance(exc, sdk.APIStatusError):
        d["kind"] = "api_error"
        d["status_code"] = exc.status_code
        d["request_id"] = getattr(exc, "request_id", None)
        body = getattr(exc, "body", None)
        d["body"] = body if isinstance(body, (dict, list, str, type(None))) else str(body)
    elif isinstance(exc, sdk.APIConnectionError):  # includes APITimeoutError
        d["kind"] = "connection_error"
    else:
        d["kind"] = "exception"
        d["traceback"] = "".join(traceback.format_exception(exc))[-4000:]
    return d


class AnthropicCaller:
    provider = "anthropic"

    def __init__(self, args):
        import anthropic

        self.sdk = anthropic
        kw = {"timeout": args.timeout, "max_retries": args.sdk_retries}
        if args.base_url:
            kw["base_url"] = args.base_url
        self.client = anthropic.Anthropic(**kw)
        self.base_url = str(self.client.base_url)
        self.args = args

    def params(self, prompt: dict) -> dict:
        p = {
            "model": self.args.model,
            "max_tokens": self.args.max_tokens,
            "system": prompt["system"],
            "messages": [{"role": "user", "content": prompt["user"]}],
        }
        if self.args.effort:
            p["output_config"] = {"effort": self.args.effort}
        return p

    def call(self, params: dict) -> dict:
        raw = self.client.messages.with_raw_response.create(**params)
        msg = raw.parse()
        return {
            "http_status": getattr(raw, "status_code", None),
            "retries_taken": getattr(raw, "retries_taken", None),
            "request_id": raw.headers.get("request-id"),
            "model_returned": msg.model,
            "stop_reason": msg.stop_reason,
            "response_text": "".join(b.text for b in msg.content if getattr(b, "type", None) == "text"),
            "usage": _to_plain(msg.usage) if msg.usage is not None else None,
            "response_raw": _to_plain(msg),
        }

    def describe_error(self, exc):
        return _describe_error(self.sdk, exc)


class OpenAICaller:
    provider = "openai"

    def __init__(self, args):
        import openai

        self.sdk = openai
        kw = {"timeout": args.timeout, "max_retries": args.sdk_retries}
        if args.base_url:
            kw["base_url"] = args.base_url
        self.client = openai.OpenAI(**kw)
        self.base_url = str(self.client.base_url)
        self.args = args

    def params(self, prompt: dict) -> dict:
        p = {
            "model": self.args.model,
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            self.args.openai_token_param: self.args.max_tokens,
        }
        if self.args.temperature is not None:
            p["temperature"] = self.args.temperature
        if self.args.effort:
            p["reasoning_effort"] = self.args.effort
        return p

    def call(self, params: dict) -> dict:
        raw = self.client.chat.completions.with_raw_response.create(**params)
        resp = raw.parse()
        choice = resp.choices[0] if resp.choices else None
        return {
            "http_status": getattr(raw, "status_code", None),
            "retries_taken": getattr(raw, "retries_taken", None),
            "request_id": raw.headers.get("x-request-id"),
            "model_returned": resp.model,
            "stop_reason": choice.finish_reason if choice else None,
            "response_text": (choice.message.content or "") if choice else "",
            "usage": _to_plain(resp.usage) if resp.usage is not None else None,
            "response_raw": _to_plain(resp),
        }

    def describe_error(self, exc):
        return _describe_error(self.sdk, exc)


class AdaptionCaller:
    """Adaption inference gateway: OpenAI Responses-style API, official `adaption` SDK.

    POST {base}/api/v1/responses with the notes as `instructions` and the
    question as `input`. store and agentic_memory are sent as false so that
    nothing is kept or carried between calls (or between conditions).
    """
    provider = "adaption"

    def __init__(self, args):
        import adaption

        self.sdk = adaption
        kw = {"timeout": args.timeout, "max_retries": args.sdk_retries}
        if args.base_url:
            kw["base_url"] = args.base_url
        self.client = adaption.Adaption(**kw)  # reads ADAPTION_API_KEY itself
        self.base_url = str(self.client.base_url)
        self.args = args

    def params(self, prompt: dict) -> dict:
        p = {
            "model": self.args.model,
            "instructions": prompt["system"],
            "input": prompt["user"],
            "max_output_tokens": self.args.max_tokens,
            "store": False,
            "agentic_memory": False,
        }
        if self.args.temperature is not None:
            p["temperature"] = self.args.temperature
        return p

    def call(self, params: dict) -> dict:
        raw = self.client.responses.with_raw_response.create(**params)
        resp = raw.parse()
        plain = _to_plain(resp)
        parts = [c for item in (plain.get("output") or []) if isinstance(item, dict) and item.get("type") == "message"
                 for c in (item.get("content") or []) if isinstance(c, dict)]
        text = getattr(resp, "output_text", None)
        if text is None:
            text = plain.get("output_text")
        if text is None:
            text = "".join(c.get("text") or "" for c in parts if c.get("type") == "output_text")
        status = plain.get("status")
        incomplete_reason = (plain.get("incomplete_details") or {}).get("reason")
        if any(c.get("type") == "refusal" for c in parts):
            stop = "refusal"
        elif status == "incomplete":
            stop = incomplete_reason or "incomplete"
        else:
            stop = status
        out = {
            "http_status": getattr(raw, "status_code", None),
            "retries_taken": getattr(raw, "retries_taken", None),
            "request_id": raw.headers.get("x-request-id") or raw.headers.get("request-id"),
            "model_returned": plain.get("model"),
            "response_status": status,
            "stop_reason": stop,
            "response_text": text or "",
            "usage": plain.get("usage"),
            "response_raw": plain,
        }
        if status not in ("completed", "incomplete"):
            # HTTP 200 but the response itself failed (or is in an unexpected state).
            err = plain.get("error") or {}
            out["call_status"] = "response_failed"
            out["call_error"] = {"kind": "response_failed", "type": f"response_status_{status}",
                                 "code": err.get("code"), "message": err.get("message") or f"response status {status!r}"}
        return out

    def describe_error(self, exc):
        return _describe_error(self.sdk, exc)


CALLERS = {"anthropic": AnthropicCaller, "openai": OpenAICaller, "adaption": AdaptionCaller}


# ---------------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------------
def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pkg_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_new_json(path: Path, obj) -> None:
    with path.open("x", encoding="utf-8") as fh:  # "x": never overwrite
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")


def append_jsonl(fh, obj) -> None:
    fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", s).strip("-")[:40] or "x"


def credential_status(provider: str) -> tuple[bool, dict]:
    present = {v: bool(os.environ.get(v)) for v in KEY_ENV_VARS[provider]}
    if any(present.values()):
        return True, present
    if provider == "anthropic" and shutil.which("ant"):
        present["ant_cli_profile_possible"] = True
        return True, present
    return False, present


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="render prompts and plan; no API calls")
    ap.add_argument("--provider", choices=sorted(CALLERS), default=os.environ.get("PILOT_PROVIDER"))
    ap.add_argument("--model", default=os.environ.get("PILOT_MODEL"), help="model id (or set PILOT_MODEL)")
    ap.add_argument("--base-url", default=None, help="override API base URL (default: SDK env var or official endpoint)")
    ap.add_argument("--cases", type=Path, default=HERE / "pilot_cases.json")
    ap.add_argument("--conditions", default="main5",
                    help="'main5' (default: the five main-pilot conditions), 'core4', 'all' (needs pilot_cases_v3.json), or a comma list")
    ap.add_argument("--case-ids", default=None, help="comma list of case ids to run (default: all cases); for smoke tests")
    ap.add_argument("--question", choices=sorted(QUESTIONS), default="probability",
                    help="'probability' (default) or 'evidence_probe' (asks whether the second note is evidence; analyse with analyze_probe.py)")
    ap.add_argument("--repeats", type=int, default=3, help="samples per case x condition")
    ap.add_argument("--max-tokens", type=int, default=4000, help="output cap; includes thinking tokens on reasoning models")
    ap.add_argument("--temperature", type=float, default=None, help="OpenAI-compatible only; omitted unless given")
    ap.add_argument("--effort", default=None, help="Anthropic output_config.effort / OpenAI reasoning_effort; omitted unless given")
    ap.add_argument("--openai-token-param", choices=["max_completion_tokens", "max_tokens"], default="max_completion_tokens")
    ap.add_argument("--seed", type=int, default=20261113, help="seed for call-order shuffle")
    ap.add_argument("--timeout", type=float, default=180.0, help="per-request timeout, seconds")
    ap.add_argument("--sdk-retries", type=int, default=2, help="SDK automatic retries on 429/5xx/connection errors")
    ap.add_argument("--abort-after", type=int, default=5, help="stop after this many consecutive failed calls")
    ap.add_argument("--max-calls", type=int, default=150, help="refuse to start if the run would make more calls")
    ap.add_argument("--max-credits", type=float, default=5.0,
                    help="refuse to start if the worst-case cost (every call using all max tokens) exceeds this many credits")
    ap.add_argument("--price-input", type=float, default=None, help="override input price, credits per 1M tokens")
    ap.add_argument("--price-output", type=float, default=None, help="override output price, credits per 1M tokens")
    ap.add_argument("--allow-unknown-price", action="store_true",
                    help="Adaption only: run even if no price is known for the model (cost then shown as tokens only)")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N planned calls (smoke test); the rest are recorded as not attempted")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    ap.add_argument("--run-id", default=None, help="explicit run folder name (must not exist)")
    ap.add_argument("--tag", default="", help="free-text label stored in the manifest")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    data, cases = load_cases(args.cases)
    problems = validate_design(cases)
    if problems:
        print("Design checks FAILED; nothing was run:", file=sys.stderr)
        for p in problems:
            print("  -", p, file=sys.stderr)
        return 3

    if args.conditions in CONDITION_SETS:
        conditions = list(CONDITION_SETS[args.conditions])
    else:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
        unknown = [c for c in conditions if c not in CONDITIONS]
        if unknown:
            print(f"Unknown conditions: {unknown}. Valid: {list(CONDITIONS)}", file=sys.stderr)
            return 2

    run_cases = cases
    if args.case_ids:
        wanted = [c.strip() for c in args.case_ids.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in {x["id"] for x in cases}]
        if unknown:
            print(f"Unknown case ids: {unknown}. Valid: {[c['id'] for c in cases]}", file=sys.stderr)
            return 2
        run_cases = [c for c in cases if c["id"] in wanted]
    unsupported = sorted({(c["id"], cond) for c in run_cases for cond in conditions if cond not in supported_conditions(c)})
    if unsupported:
        print(f"These cases lack text for the requested conditions: {unsupported[:4]}{' ...' if len(unsupported) > 4 else ''}. "
              "Use --cases pilot_cases_v3.json for disconfirming_evidence.", file=sys.stderr)
        return 2

    plan, order = build_plan(run_cases, conditions, args.repeats, args.seed)
    to_run = order if args.limit is None else order[: args.limit]

    caller = None
    cred_present = {}
    if not args.dry_run:
        if not args.provider or not args.model:
            print("A live run needs --provider and --model (or PILOT_PROVIDER and PILOT_MODEL). "
                  "Use --dry-run to check prompts without calling a model.", file=sys.stderr)
            return 2
        if args.provider == "anthropic" and args.temperature is not None:
            print("--temperature is not sent to Anthropic: current Claude models and SDKs reject sampling parameters.", file=sys.stderr)
            return 2
        if args.provider == "adaption" and args.effort:
            print("--effort is not sent to Adaption: its responses API documents no reasoning-effort parameter.", file=sys.stderr)
            return 2
        if len(to_run) > args.max_calls:
            print(f"Run would make {len(to_run)} calls, above --max-calls {args.max_calls}. Raise --max-calls deliberately to proceed.", file=sys.stderr)
            return 2
        ok, cred_present = credential_status(args.provider)
        if not ok:
            names = " or ".join(KEY_ENV_VARS[args.provider])
            print(f"No credentials for provider '{args.provider}': set {names} in your environment. Nothing was run.", file=sys.stderr)
            return 2
        try:
            caller = CALLERS[args.provider](args)
        except Exception as e:  # e.g. SDK not installed, client could not resolve credentials
            print(f"Could not create the {args.provider} client: {type(e).__name__}: {e}", file=sys.stderr)
            return 2

    # Pricing and worst-case cost, checked before any folder is created.
    price, price_source = None, None
    if not args.dry_run:
        price = PRICES.get((args.provider, args.model))
        price_source = PRICE_SOURCE.get(args.provider) if price else None
        if args.price_input is not None or args.price_output is not None:
            if args.price_input is None or args.price_output is None:
                print("Give both --price-input and --price-output.", file=sys.stderr)
                return 2
            price = {"input": args.price_input, "cached_input": args.price_input, "output": args.price_output}
            price_source = "command line (--price-input/--price-output)"
        if price is None and args.provider == "adaption" and not args.allow_unknown_price:
            print(f"No price known for Adaption model '{args.model}'. Check docs.adaptionlabs.ai/inference-stacks/models and "
                  "pass --price-input/--price-output, or --allow-unknown-price to run with token counts only.", file=sys.stderr)
            return 2
    est_in_tokens = sum(
        math.ceil(len(render(c, cond)["system"] + render(c, cond)["user"]) / 3)  # ~4 chars/token; /3 is conservative
        for c in run_cases for cond in conditions) * args.repeats * (len(to_run) / max(len(plan), 1))
    worst_case = None
    if price:
        worst_case = (est_in_tokens * price["input"] + len(to_run) * args.max_tokens * price["output"]) / 1e6
        if worst_case > args.max_credits:
            print(f"Worst-case cost {worst_case:.3f} credits exceeds --max-credits {args.max_credits}. "
                  "Raise it deliberately or lower --max-tokens/--repeats.", file=sys.stderr)
            return 2

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode = "dryrun" if args.dry_run else f"{args.provider}_{slug(args.model)}"
    run_id = args.run_id or f"{stamp}_{mode}_{secrets.token_hex(2)}"
    run_dir = args.out_dir / run_id
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()  # fails if it exists: previous results are never overwritten
    except FileExistsError:
        print(f"Refusing to overwrite existing run folder: {run_dir}", file=sys.stderr)
        return 4

    lengths = length_table(cases)
    manifest = {
        "run_id": run_id,
        "created_at": utcnow(),
        "mode": "dry_run" if args.dry_run else "live",
        "tag": args.tag,
        "argv": sys.argv if argv is None else ["pilot.py", *argv],
        "template_version": TEMPLATE_VERSION,
        "templates": {
            "system": SYSTEM_TEMPLATE, "user": USER_TEMPLATE, "slot_a": SLOT_A, "slot_b": SLOT_B,
            "conditions": {k: list(v) for k, v in CONDITIONS.items()},
        },
        "settings": {
            "provider": args.provider if not args.dry_run else None,
            "model_requested": args.model if not args.dry_run else None,
            "base_url": caller.base_url if caller else None,
            "conditions": conditions,
            "repeats": args.repeats,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature if args.provider in ("openai", "adaption") else None,
            "temperature_note": "not sent; provider default" if args.temperature is None else "sent",
            "effort": args.effort,
            "effort_note": "not sent; provider default" if not args.effort else "sent",
            "openai_token_param": args.openai_token_param if args.provider == "openai" else None,
            "adaption_store": False if args.provider == "adaption" else None,
            "adaption_agentic_memory": False if args.provider == "adaption" else None,
            "case_ids_run": [c["id"] for c in run_cases],
            "pricing": {
                "credits_per_1M": price,
                "source": price_source,
                "est_input_tokens_planned": round(est_in_tokens) if not args.dry_run else None,
                "worst_case_credits": worst_case,
                "max_credits": args.max_credits,
            },
            "order_seed": args.seed,
            "timeout_s": args.timeout,
            "sdk_retries": args.sdk_retries,
            "abort_after": args.abort_after,
            "limit": args.limit,
            "max_calls": args.max_calls,
        },
        "credentials_env_present": cred_present,  # booleans only; values are never read into output
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "anthropic_sdk": pkg_version("anthropic"),
            "openai_sdk": pkg_version("openai"),
            "base_url_env_set": {v: bool(os.environ.get(v)) for v in BASE_URL_ENV_VARS.values()},
        },
        "inputs": {
            "cases_file": str(args.cases),
            "cases_sha256": sha256_file(args.cases),
            "pilot_py_sha256": sha256_file(Path(__file__)),
            "n_cases": len(cases),
            "case_ids": [c["id"] for c in cases],
            "case_ids_run": [c["id"] for c in run_cases],
        },
        "design_checks": {"passed": True, "problems": []},
        "lengths": {"per_condition": lengths["per_condition"], "attributed_vs_unlabelled": lengths["attributed_vs_unlabelled"]},
        "plan": plan,
        "execution_order": order,
        "n_planned": len(plan),
        "n_to_attempt": len(to_run),
    }
    write_new_json(run_dir / "manifest.json", manifest)
    with (run_dir / "prompts.jsonl").open("x", encoding="utf-8") as fh:
        for case in cases:
            for c in supported_conditions(case):
                v = render(case, c)
                append_jsonl(fh, {"case_id": case["id"], "condition": c, "slot_a": CONDITIONS[c][0], "slot_b": CONDITIONS[c][1], **v})

    print(f"Run folder: {run_dir}")
    print(f"Design checks passed for {len(cases)} cases.")
    print("Prompt length by condition (system + user, mean over cases):")
    for c, s in lengths["per_condition"].items():
        print(f"  {c:34s} {s['mean_chars']:7.1f} chars  {s['mean_words']:5.1f} words")
    d = lengths["attributed_vs_unlabelled"]
    print("Attributed minus unlabelled, per case: chars "
          f"{sorted({x['attributed_minus_unlabelled_chars'] for x in d})}, words "
          f"{sorted({x['attributed_minus_unlabelled_words'] for x in d})}")
    print(f"Plan: {len(run_cases)} cases x {len(conditions)} conditions x {args.repeats} repeats = {len(plan)} calls; "
          f"will attempt {0 if args.dry_run else len(to_run)}.")
    if not args.dry_run:
        print(f"Endpoint: {caller.base_url}  model: {args.model}")
        if price:
            print(f"Price ({price_source}): {price} credits per 1M tokens. "
                  f"Worst case if every call used all {args.max_tokens} output tokens: {worst_case:.4f} credits "
                  f"(cap --max-credits {args.max_credits}).")
        else:
            print("Price unknown: cost will be reported as token counts only.")

    if args.dry_run:
        ex = cases[0]
        print(f"\nExample prompts for case '{ex['id']}':")
        for c in supported_conditions(ex):
            v = render(ex, c)
            print(f"\n=== {c} ===\n[system]\n{v['system']}\n[user]\n{v['user']}")
        write_new_json(run_dir / "run_end.json", {"status": "dry_run", "finished_at": utcnow(), "counts": {}})
        return 0

    # ---------------- live run ----------------
    def _sigterm(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, _sigterm)

    prompts = {(case["id"], c): render(case, c) for case in run_cases for c in conditions}
    counts: Counter = Counter()
    usage_tot: Counter = Counter()
    credits_tot = 0.0
    consecutive_fail = 0
    end = {"status": "crashed"}
    in_flight = None
    with (run_dir / "calls.jsonl").open("x", encoding="utf-8") as fh:
        try:
            for k, pi in enumerate(to_run):
                entry = plan[pi]
                params = caller.params(prompts[(entry["case_id"], entry["condition"])])
                rec = {
                    "record_type": "call", "run_id": run_id, "call_index": k, **entry,
                    "provider": caller.provider, "model_requested": args.model, "base_url": caller.base_url,
                    "request": params, "started_at": utcnow(),
                }
                in_flight = entry
                t0 = time.monotonic()
                try:
                    out = caller.call(params)
                    rec["status"] = out.pop("call_status", "ok")
                    rec["error"] = out.pop("call_error", None)
                    rec.update(out)
                except Exception as e:  # recorded, never swallowed silently
                    err = caller.describe_error(e)
                    rec["status"] = err["kind"]
                    rec["error"] = err
                rec["latency_s"] = round(time.monotonic() - t0, 3)
                rec["finished_at"] = utcnow()
                append_jsonl(fh, rec)
                in_flight = None
                counts[rec["status"]] += 1
                norm = normalise_usage(rec.get("usage"))
                for key_, v in norm.items():
                    usage_tot[key_] += v or 0
                credits_tot += estimate_credits(norm, price) or 0.0
                consecutive_fail = 0 if rec["status"] == "ok" else consecutive_fail + 1
                extra = "" if rec["status"] == "ok" else f"  [{rec['error'].get('type')}: {str(rec['error'].get('message'))[:120]}]"
                cost = f" | est. {credits_tot:.4f} credits" if price else f" | {usage_tot['output_tokens']} output tokens"
                print(f"[{k + 1}/{len(to_run)}] {entry['case_id']} / {entry['condition']} / r{entry['repeat']} -> "
                      f"{rec['status']} ({rec['latency_s']:.1f}s){cost}{extra}", flush=True)
                if consecutive_fail >= args.abort_after:
                    end = {"status": "aborted", "reason": f"{consecutive_fail} consecutive failed calls"}
                    break
            else:
                end = {"status": "limited" if args.limit is not None else "completed"}
        except KeyboardInterrupt as e:
            end = {"status": "interrupted", "reason": str(e) or "KeyboardInterrupt"}
        except BaseException as e:
            end = {"status": "crashed", "reason": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
            raise
        finally:
            end.update({
                "finished_at": utcnow(),
                "counts": dict(counts),
                "n_recorded": sum(counts.values()),
                "n_to_attempt": len(to_run),
                "n_planned": len(plan),
                "call_in_flight_when_stopped": in_flight,
                "usage_totals": dict(usage_tot),
                "est_credits": round(credits_tot, 6) if price else None,
                "est_credits_note": "estimate from reported usage x price in manifest; not the billed amount",
            })
            write_new_json(run_dir / "run_end.json", end)
            print(f"\nRun {end['status']}: {dict(counts)}. Raw records in {run_dir / 'calls.jsonl'}")
            print(f"Tokens: {dict(usage_tot)}" + (f"; estimated {credits_tot:.4f} credits" if price else ""))
            print(f"Analyse with: python analyze.py {run_dir}")
    return 0 if end["status"] in ("completed", "limited") else 1


if __name__ == "__main__":
    sys.exit(main())
