#!/usr/bin/env python3
"""Analyse one pilot run folder, or combine several.

    python analyze.py results/<run_id>
    python analyze.py results/<run_a> results/<run_b>     # combined analysis

Reads manifest.json, prompts.jsonl, calls.jsonl and run_end.json; never
modifies them. A single run writes <run_id>/analysis-<UTC timestamp>/. A
combined analysis writes results/combined/<UTC timestamp>_<n>runs/. Either
way the output is report.md plus CSVs, and the report is printed.

Combining is refused unless the runs share provider and model, and every
case x condition that appears in more than one run has byte-identical prompts.
Repeated conditions get a between-run drift check. Valid responses are pooled
per case x condition.

Unit of analysis: the case. Repeats are averaged within each case x condition
cell first, then contrasts are computed per case and summarised across cases.
Every planned call appears in parsed_calls.csv, including failed, unparseable
and never-attempted ones.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

VALID = "ok"
REFUSAL_STOPS = {"refusal", "content_filter"}
TRUNCATION_STOPS = {"max_tokens", "length", "max_output_tokens", "incomplete"}
P_CLIP = (1.0, 99.0)  # for log-odds; stated probabilities of 0 or 100 are clipped here
BOOT_N, BOOT_SEED = 10000, 12345

# Pre-specified contrasts: weights over condition cell means, computed per case.
CONTRASTS = [
    ("carryover_unlabelled", {"unlabelled_summary": 1, "observation_only": -1},
     "Unlabelled summary minus observation only. >0 means the carried-over interpretation raised stated probability."),
    ("carryover_attributed", {"attributed_summary": 1, "observation_only": -1},
     "Attributed summary minus observation only."),
    ("attribution_reduction", {"unlabelled_summary": 1, "attributed_summary": -1},
     "Unlabelled minus attributed. >0 means attribution reduced the carry-over."),
    ("evidence_effect", {"new_evidence": 1, "observation_only": -1},
     "New evidence minus observation only. Manipulation check: must be clearly >0."),
    ("evidence_effect_under_attribution", {"attributed_summary_plus_evidence": 1, "attributed_summary": -1},
     "Attributed + evidence minus attributed. Response to new evidence when the interpretation is attributed."),
    ("evidence_preservation", {"attributed_summary_plus_evidence": 1, "attributed_summary": -1, "new_evidence": -1, "observation_only": 1},
     "Evidence effect under attribution minus evidence effect alone. About 0 means responsiveness preserved; <0 means attribution blunted updating."),
    # Added 2026-11-15, before any data for these conditions existed (README, "Extension").
    ("disconfirming_effect", {"disconfirming_evidence": 1, "observation_only": -1},
     "Disconfirming evidence minus observation only. Second manipulation check: must be clearly <0, i.e. the model also updates downward."),
    ("interpretation_effect_with_evidence", {"unlabelled_summary_plus_evidence": 1, "new_evidence": -1},
     "Unlabelled summary + evidence minus evidence alone. >0 means the carried-over interpretation still raises the estimate when new evidence is present."),
    ("attribution_reduction_with_evidence", {"unlabelled_summary_plus_evidence": 1, "attributed_summary_plus_evidence": -1},
     "Unlabelled + evidence minus attributed + evidence. >0 means attribution reduced the carry-over when evidence is present; <0 means the label increased it."),
    # Added 2026-11-15, before any perseverance-run data existed (README, "Perseverance run").
    ("perseverance_unlabelled", {"unlabelled_summary_plus_disconfirming": 1, "disconfirming_evidence": -1},
     "Unlabelled summary + disconfirming evidence minus disconfirming evidence alone. >0 means the carried-over guess kept the estimate up against contrary evidence."),
    ("perseverance_attributed", {"attributed_summary_plus_disconfirming": 1, "disconfirming_evidence": -1},
     "Attributed summary + disconfirming evidence minus disconfirming evidence alone."),
    ("attribution_reduction_with_disconfirming", {"unlabelled_summary_plus_disconfirming": 1, "attributed_summary_plus_disconfirming": -1},
     "Unlabelled + disconfirming minus attributed + disconfirming. >0 means attribution reduced perseverance."),
    # Added 2026-11-15, before any label-strength data existed (README, "Label-strength run").
    ("carryover_asserted", {"asserted_summary": 1, "observation_only": -1},
     "Guess stated as a plain note (no source, no hedge) minus observation only."),
    ("carryover_warned", {"warned_summary": 1, "observation_only": -1},
     "Guess labelled 'as an unverified guess' minus observation only."),
    ("warning_reduction", {"unlabelled_summary": 1, "warned_summary": -1},
     "Unlabelled minus warned. >0 means the unverified-guess label reduced the carry-over."),
    ("hedge_removal_effect", {"asserted_summary": 1, "unlabelled_summary": -1},
     "Plain-note guess minus hedged unlabelled guess. >0 means dropping 'It was suggested' raised the carry-over."),
    ("source_user_vs_assistant", {"user_attributed_summary": 1, "attributed_summary": -1},
     "'The user suggested' minus 'The assistant suggested' (minimal pair). >0 means a user's suggestion counts for more."),
]
GATES = [  # (contrast, required sign, description)
    ("evidence_effect", +1, "mean > 0 and bootstrap CI lower bound > 0"),
    ("disconfirming_effect", -1, "mean < 0 and bootstrap CI upper bound < 0"),
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_NUM = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*%?\s*$")


def _json_objects(text: str):
    dec = json.JSONDecoder()
    i = 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        yield obj, i, end
        i = end


def parse_response(text, stop_reason) -> dict:
    """Return {status, probability, reason, flags}. status 'ok' is the only one analysed."""
    flags: list[str] = []
    if stop_reason in REFUSAL_STOPS:
        return {"status": "refusal", "probability": None, "reason": None, "flags": flags}
    if not text or not text.strip():
        # e.g. a reasoning model that spent the whole output budget thinking
        status = "truncated" if stop_reason in TRUNCATION_STOPS else "empty"
        return {"status": status, "probability": None, "reason": None, "flags": flags}
    found = [(o, s, e) for o, s, e in _json_objects(text) if isinstance(o, dict) and "probability" in o]
    if not found:
        status = "truncated" if stop_reason in TRUNCATION_STOPS else "no_json"
        return {"status": status, "probability": None, "reason": None, "flags": flags}
    if stop_reason in TRUNCATION_STOPS:
        flags.append("stopped_at_token_limit")
    obj, s, e = found[0]
    outside = (text[:s] + text[e:]).strip().strip("`").strip()
    if outside and outside.lower() != "json":
        flags.append("text_outside_json")
    elif outside:
        flags.append("code_fence")

    def num(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            m = _NUM.match(v)
            return float(m.group(1)) if m else None
        return None

    values = {num(o["probability"]) for o, _, _ in found}
    if len(values) > 1:
        return {"status": "conflicting_json", "probability": None, "reason": None, "flags": flags}
    p = num(obj["probability"])
    reason = obj.get("reason") if isinstance(obj.get("reason"), str) else None
    if p is None or math.isnan(p):
        return {"status": "not_a_number", "probability": None, "reason": reason, "flags": flags}
    if not 0 <= p <= 100:
        return {"status": "out_of_range", "probability": p, "reason": reason, "flags": flags}
    if 0 < p < 1 and p != int(p):
        # Probably answered on a 0-1 scale despite the instruction; do not guess.
        return {"status": "ambiguous_scale", "probability": p, "reason": reason, "flags": flags}
    if p != int(p):
        flags.append("non_integer")
    return {"status": VALID, "probability": p, "reason": reason, "flags": flags}


# ---------------------------------------------------------------------------
# Usage and cost (shared with pilot.py for its live running total)
# ---------------------------------------------------------------------------
def normalise_usage(usage) -> dict:
    """Map Anthropic, OpenAI chat and Responses-API usage objects to one shape.

    Cached and reasoning tokens are breakdowns: cached_input_tokens is a subset
    of input_tokens and reasoning_tokens a subset of output_tokens, as in the
    OpenAI Responses convention that Adaption's API follows.
    """
    if not isinstance(usage, dict):
        return {}
    inp = usage.get("input_tokens", usage.get("prompt_tokens"))
    out = usage.get("output_tokens", usage.get("completion_tokens"))
    in_det = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    out_det = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": inp,
        "cached_input_tokens": (in_det.get("cached_tokens") if isinstance(in_det, dict) else None) or 0,
        "output_tokens": out,
        "reasoning_tokens": (out_det.get("reasoning_tokens") if isinstance(out_det, dict) else None) or 0,
    }


def estimate_credits(norm: dict, price: dict | None) -> float | None:
    """Credits for one call from normalised usage and a per-1M-token price. None if unknown."""
    if not price or norm.get("input_tokens") is None or norm.get("output_tokens") is None:
        return None
    cached = min(norm.get("cached_input_tokens") or 0, norm["input_tokens"])
    return ((norm["input_tokens"] - cached) * price["input"]
            + cached * price.get("cached_input", price["input"])
            + norm["output_tokens"] * price["output"]) / 1e6


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def logit_pct(p: float) -> float:
    q = min(max(p, P_CLIP[0]), P_CLIP[1]) / 100.0
    return math.log(q / (1 - q))


def sign_test_p(values: list[float]) -> float | None:
    pos = sum(v > 0 for v in values)
    neg = sum(v < 0 for v in values)
    n = pos + neg
    if n == 0:
        return None
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def boot_ci(values: list[float], seed: int = BOOT_SEED) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(BOOT_N))
    return means[int(0.025 * BOOT_N)], means[int(0.975 * BOOT_N) - 1]


def summarise(values: list[float]) -> dict:
    ci = boot_ci(values)
    return {
        "n_cases": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "ci95_low": ci[0] if ci else None,
        "ci95_high": ci[1] if ci else None,
        "n_pos": sum(v > 0 for v in values),
        "n_neg": sum(v < 0 for v in values),
        "n_zero": sum(v == 0 for v in values),
        "sign_test_p": sign_test_p(values),
    }


def fmt(x, nd=1):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:+.{nd}f}" if nd is not None else f"{x}"
    return str(x)


def fnum(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def load_run(run_dir: Path):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    calls = []
    bad_lines = []
    cpath = run_dir / "calls.jsonl"
    if cpath.exists():
        for ln, line in enumerate(cpath.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                calls.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad_lines.append((ln, str(e)))  # e.g. a half-written final line after a hard kill
    rpath = run_dir / "run_end.json"
    run_end = json.loads(rpath.read_text(encoding="utf-8")) if rpath.exists() else None
    return manifest, calls, run_end, bad_lines


def load_prompts(run_dir: Path) -> dict:
    out = {}
    p = run_dir / "prompts.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                out[(o["case_id"], o["condition"])] = (o["system"], o["user"])
    return out


class IncompatibleRuns(Exception):
    pass


def check_compatible(runs: list[dict]) -> list[str]:
    """Reasons the runs cannot be pooled; empty if they can."""
    problems = []
    base = runs[0]["manifest"]
    for r in runs[1:]:
        m = r["manifest"]
        for k in ("provider", "model_requested"):
            if base["settings"].get(k) != m["settings"].get(k):
                problems.append(f"{k} differs: {base['settings'].get(k)!r} vs {m['settings'].get(k)!r}")
        if base.get("template_version") != m.get("template_version"):
            problems.append(f"template_version differs: {base.get('template_version')} vs {m.get('template_version')}")
    seen: dict = {}
    for r in runs:
        rid = r["manifest"]["run_id"]
        ran = set(r["manifest"]["settings"]["conditions"])
        for key, prompt in r["prompts"].items():
            if key[1] not in ran:
                continue
            if key in seen and seen[key][1] != prompt:
                problems.append(f"prompt for {key[0]}/{key[1]} differs between {seen[key][0]} and {rid}")
            seen.setdefault(key, (rid, prompt))
    return problems


def _row(entry: dict, rec: dict, n_recs: int, price) -> dict:
    row = {
        **entry, "call_index": rec.get("call_index"), "call_status": rec["status"],
        "model_returned": rec.get("model_returned"), "stop_reason": rec.get("stop_reason"),
        "latency_s": rec.get("latency_s"), "retries_taken": rec.get("retries_taken"),
        "request_id": rec.get("request_id"),
        "duplicate_record": n_recs > 1,
    }
    if rec["status"] == "ok":
        pr = parse_response(rec.get("response_text"), rec.get("stop_reason"))
        row.update(outcome=pr["status"], probability=pr["probability"], reason=pr["reason"],
                   parse_flags=";".join(pr["flags"]),
                   response_text_head=(rec.get("response_text") or "")[:300])
    else:
        err = rec.get("error") or {}
        row.update(outcome=rec["status"], error_type=err.get("type"),
                   error_status_code=err.get("status_code") or err.get("code"),
                   error_message=str(err.get("message"))[:300])
    norm = normalise_usage(rec.get("usage"))
    row.update({k: norm.get(k) for k in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")})
    row["est_credits"] = estimate_credits(norm, price)
    return row


def _cell(values: list[float]) -> dict:
    return {
        "n_valid": len(values),
        "mean_pp": statistics.fmean(values) if values else None,
        "sd_pp": statistics.stdev(values) if len(values) > 1 else None,
        "mean_logodds": statistics.fmean(logit_pct(x) for x in values) if values else None,
        "values": values,
    }


def analyse(run_dirs) -> tuple[str, dict]:
    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]
    runs = []
    for d in run_dirs:
        d = Path(d)
        m, calls, end, bad = load_run(d)
        runs.append({"dir": d, "manifest": m, "calls": calls, "run_end": end, "bad_lines": bad, "prompts": load_prompts(d)})
    if len(runs) > 1:
        problems = check_compatible(runs)
        if problems:
            raise IncompatibleRuns(problems)
    multi = len(runs) > 1
    manifest = runs[0]["manifest"]  # reference run for settings shared by all runs
    s = manifest["settings"]
    run_ids = [r["manifest"]["run_id"] for r in runs]

    conditions, case_ids, plan = [], [], []
    for r in runs:
        m = r["manifest"]
        conditions += [c for c in m["settings"]["conditions"] if c not in conditions]
        case_ids += [c for c in m["inputs"].get("case_ids_run", m["inputs"]["case_ids"]) if c not in case_ids]
        plan += [{**e, "run_id": m["run_id"]} for e in m["plan"]]

    # One row per planned call. If a plan entry was somehow recorded twice, keep all records but flag it.
    rows = []
    for r in runs:
        rid = r["manifest"]["run_id"]
        price = (r["manifest"]["settings"].get("pricing") or {}).get("credits_per_1M")
        by_plan = defaultdict(list)
        for c in r["calls"]:
            by_plan[c["plan_index"]].append(c)
        for entry in r["manifest"]["plan"]:
            entry = {**entry, "run_id": rid}
            recs = by_plan.get(entry["plan_index"], [])
            if not recs:
                rows.append({**entry, "call_status": "not_attempted", "outcome": "not_attempted"})
            for rec in recs:
                rows.append(_row(entry, rec, len(recs), price))

    # Flags about provenance of the data itself.
    models_returned = Counter(r.get("model_returned") for r in rows if r.get("model_returned"))
    hosts = sorted({urlparse(r["manifest"]["settings"].get("base_url") or "").hostname or "" for r in runs})
    host = ", ".join(h for h in hosts if h)
    synthetic = any(h in {"127.0.0.1", "localhost", "::1"} for h in hosts) or any("mock" in (m or "").lower() for m in models_returned)
    dry = all(r["manifest"].get("mode") == "dry_run" for r in runs)

    # Cells: case x condition, pooled over runs; also kept per run for the drift check.
    pooled, per_run = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["outcome"] == VALID and not r.get("duplicate_record"):
            pooled[(r["case_id"], r["condition"])].append(r["probability"])
            per_run[(r["run_id"], r["case_id"], r["condition"])].append(r["probability"])
    planned_per_cell = Counter((e["case_id"], e["condition"]) for e in plan)
    cells = {key: {"case_id": key[0], "condition": key[1], "n_planned": n, **_cell(pooled.get(key, []))}
             for key, n in planned_per_cell.items()}

    # Contrasts per case.
    contrast_results, per_case_contrasts = [], []
    for name, weights, desc in CONTRASTS:
        if not all(c in conditions for c in weights):
            contrast_results.append({"contrast": name, "description": desc, "skipped": "condition not in this run"})
            continue
        pp_vals, lo_vals, dropped = [], [], []
        for cid in case_ids:
            cs = [cells.get((cid, c)) for c in weights]
            if any(c is None or c["n_valid"] == 0 for c in cs):
                dropped.append(cid)
                continue
            pp = sum(w * cells[(cid, c)]["mean_pp"] for c, w in weights.items())
            lo = sum(w * cells[(cid, c)]["mean_logodds"] for c, w in weights.items())
            pp_vals.append(pp)
            lo_vals.append(lo)
            per_case_contrasts.append({"contrast": name, "case_id": cid, "value_pp": pp, "value_logodds": lo})
        contrast_results.append({"contrast": name, "description": desc, "dropped_cases": dropped,
                                 "pp": summarise(pp_vals), "logodds": summarise(lo_vals)})

    # Drift: conditions run in more than one run, compared with the first run that has them.
    drift = []
    if multi:
        for cond in conditions:
            having = [rid for rid in run_ids if any(e["condition"] == cond and e["run_id"] == rid for e in plan)]
            for later in having[1:]:
                ref = having[0]
                pp_d, lo_d = [], []
                for cid in case_ids:
                    a, b = per_run.get((ref, cid, cond)), per_run.get((later, cid, cond))
                    if a and b:
                        ca, cb = _cell(a), _cell(b)
                        pp_d.append(cb["mean_pp"] - ca["mean_pp"])
                        lo_d.append(cb["mean_logodds"] - ca["mean_logodds"])
                drift.append({"condition": cond, "reference_run": ref, "later_run": later,
                              "pp": summarise(pp_d), "logodds": summarise(lo_d)})

    # ------------------------------------------------------------------ report
    L = []
    L.append(f"# Combined analysis: {' + '.join(run_ids)}" if multi else f"# Pilot analysis: {manifest['run_id']}")
    L.append(f"\nGenerated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} by analyze.py from the raw files in "
             + ("these run folders" if multi else "this run folder") + ".\n")
    if synthetic:
        L.append("> **SYNTHETIC DATA. These are not model results.** This run used a local endpoint or a model id "
                 "containing 'mock'. Its numbers only test the pipeline.\n")
    if dry:
        L.append("> Dry run: prompts and plan only, no calls were made.\n")
    n_valid = sum(r["outcome"] == VALID for r in rows)
    n_planned = len(plan)
    for r in runs:
        tag = f" ({r['manifest']['run_id']})" if multi else ""
        end = r["run_end"]
        if end is None:
            L.append(f"> **run_end.json is missing{tag}.** The run crashed, was killed, or is still running. Results are incomplete.\n")
        elif end["status"] not in ("completed", "dry_run"):
            reason = f" {end['reason']}." if end.get("reason") else ""
            L.append(f"> **Run status{tag}: {end['status']}.**{reason} Results are incomplete.\n")
        if r["bad_lines"]:
            L.append(f"> {len(r['bad_lines'])} unreadable line(s) in calls.jsonl{tag}: {r['bad_lines']}\n")
    if n_valid < n_planned and not dry:
        L.append(f"> {n_planned - n_valid} of {n_planned} planned calls have no valid probability (see Completeness and Failures).\n")

    L.append("## Run\n" if not multi else "## Runs\n")
    if dry:
        source = "none (dry run)"
    elif synthetic:
        source = f"SYNTHETIC local mock at {host or '?'}; not model output"
    else:
        source = f"live API responses from {host or '?'}"
    shared = [
        ["Data source", source],
        ["Provider", s.get("provider")],
        ["Model requested", s.get("model_requested")],
        ["Model(s) returned", ", ".join(f"{m} ({n})" for m, n in models_returned.items()) or "–"],
        ["Base URL", s.get("base_url")],
        ["Conditions", ", ".join(conditions)],
    ]
    if not multi:
        end = run_end = runs[0]["run_end"]
        L.append(md_table(["Field", "Value"], shared + [
            ["Repeats per cell", s.get("repeats")],
            ["max_tokens", s.get("max_tokens")],
            ["Temperature", f"{s.get('temperature')} ({s.get('temperature_note')})"],
            ["Effort", f"{s.get('effort')} ({s.get('effort_note')})"],
            ["SDK retries / timeout", f"{s.get('sdk_retries')} / {s.get('timeout_s')} s"],
            ["Template version", manifest.get("template_version")],
            ["Cases file sha256", manifest["inputs"]["cases_sha256"][:16] + "…"],
            ["Created", manifest.get("created_at")],
            ["Run status", (end["status"] if end else "MISSING") + (f" ({end.get('reason')})" if end and end.get("reason") else "")],
            ["Call in flight when stopped", (end or {}).get("call_in_flight_when_stopped") or "–"],
        ]))
    else:
        L.append(md_table(["Field", "Value"], shared + [["Template version", manifest.get("template_version")],
                                                        ["Compatibility", "provider, model, template and all shared prompts identical"]]))
        L.append("")
        L.append(md_table(["Run", "Created", "Status", "Conditions", "Repeats", "max_tokens", "Temperature", "Cases file sha256"], [
            [r["manifest"]["run_id"], r["manifest"].get("created_at"), (r["run_end"] or {}).get("status", "MISSING"),
             ", ".join(r["manifest"]["settings"]["conditions"]), r["manifest"]["settings"].get("repeats"),
             r["manifest"]["settings"].get("max_tokens"), r["manifest"]["settings"].get("temperature_note"),
             r["manifest"]["inputs"]["cases_sha256"][:16] + "…"] for r in runs]))
    if models_returned and s.get("model_requested") and set(models_returned) != {s.get("model_requested")}:
        L.append("\nNote: the returned model id differs from the requested one. Check whether this is an alias or a substitution.")

    # Completeness
    L.append("\n## Completeness\n")
    outcome_kinds = sorted({r["outcome"] for r in rows})
    comp_rows = []
    for cond in conditions:
        sel = [r for r in rows if r["condition"] == cond]
        oc = Counter(r["outcome"] for r in sel)
        comp_rows.append([cond, len([e for e in plan if e["condition"] == cond]),
                          sum(r["outcome"] != "not_attempted" for r in sel)] + [oc.get(k, 0) for k in outcome_kinds])
    L.append(md_table(["Condition", "Planned", "Attempted"] + outcome_kinds, comp_rows))
    retried = [r for r in rows if (r.get("retries_taken") or 0) > 0]
    if retried:
        L.append(f"\n{len(retried)} call(s) succeeded only after automatic SDK retries.")

    # Calls and cost
    attempted = [r for r in rows if r["outcome"] != "not_attempted"]
    if attempted:
        def tot(k):
            return sum(r.get(k) or 0 for r in attempted)
        prices = {json.dumps((r["manifest"]["settings"].get("pricing") or {}).get("credits_per_1M")) for r in runs}
        sources = {(r["manifest"]["settings"].get("pricing") or {}).get("source") or "–" for r in runs}
        credits = [r["est_credits"] for r in attempted if r.get("est_credits") is not None]
        with_usage = sum(r.get("input_tokens") is not None for r in attempted)
        L.append("\n## Calls and cost\n")
        L.append(md_table(["Item", "Value"], [
            ["Calls attempted / planned", f"{len(attempted)} / {n_planned}"],
            ["HTTP requests incl. SDK retries", len(attempted) + sum(r.get("retries_taken") or 0 for r in attempted)],
            ["Calls reporting token usage", with_usage],
            ["Input tokens (of which cached)", f"{tot('input_tokens')} ({tot('cached_input_tokens')})"],
            ["Output tokens (of which reasoning)", f"{tot('output_tokens')} ({tot('reasoning_tokens')})"],
            ["Price used (credits per 1M tokens)", "; ".join(p if p != "null" else "unknown" for p in sorted(prices))],
            ["Price source", "; ".join(sorted(sources))],
            ["Estimated credits", f"{sum(credits):.4f}" if credits else "not estimated (no price)"],
        ]))
        if credits:
            L.append("\nEstimated from reported token usage and the price above. Reasoning tokens are assumed to be "
                     "included in output tokens and billed at the output rate. Check your provider dashboard for the "
                     "billed amount.")

    # Failures
    fails = [r for r in rows if r["outcome"] not in (VALID, "not_attempted")]
    L.append(f"\n## Failures and unusable responses ({len(fails)})\n")
    if fails:
        L.append(md_table((["Run"] if multi else []) + ["Plan #", "Case", "Condition", "Rep", "Outcome", "Detail"], [
            ([r["run_id"]] if multi else []) + [r["plan_index"], r["case_id"], r["condition"], r["repeat"], r["outcome"],
             (f"{r.get('error_type')} {r.get('error_status_code') or ''}: {r.get('error_message')}" if r["call_status"] != "ok"
              else f"stop={r.get('stop_reason')}; text: {r.get('response_text_head', '')[:160]!r}")]
            for r in fails]))
    else:
        L.append("None.")
    not_att = [r for r in rows if r["outcome"] == "not_attempted"]
    if not_att and not dry:
        L.append(f"\n{len(not_att)} planned call(s) were never attempted; they are listed in parsed_calls.csv.")
    flagged = [r for r in rows if r["outcome"] == VALID and r.get("parse_flags")]
    if flagged:
        L.append(f"\n{len(flagged)} valid response(s) needed lenient parsing (flags: "
                 f"{dict(Counter(f for r in flagged for f in r['parse_flags'].split(';')))}).")

    data = {"rows": rows, "cells": cells, "per_case_contrasts": per_case_contrasts, "contrasts": contrast_results,
            "drift": drift, "run_ids": run_ids}
    if n_valid == 0:
        L.append("\n## Results\n\nNo valid responses, so there is nothing to analyse.")
        return "\n".join(L) + "\n", data

    # Between-run drift
    if multi:
        L.append("\n## Between-run drift on repeated conditions\n")
        if drift:
            L.append("Per-case mean in the later run minus the earlier run. Values near 0 support pooling the runs.\n")
            L.append(md_table(["Condition", "Earlier run", "Later run", "Cases", "Mean pp [95% CI]", "Mean log-odds [95% CI]", "+/−/0"], [
                [d["condition"], d["reference_run"], d["later_run"], d["pp"]["n_cases"],
                 f"{fmt(d['pp']['mean'])} [{fmt(d['pp']['ci95_low'])}, {fmt(d['pp']['ci95_high'])}]",
                 f"{fmt(d['logodds']['mean'], 2)} [{fmt(d['logodds']['ci95_low'], 2)}, {fmt(d['logodds']['ci95_high'], 2)}]",
                 f"{d['pp']['n_pos']}/{d['pp']['n_neg']}/{d['pp']['n_zero']}"] for d in drift]))
        else:
            L.append("No condition was run in more than one run, so drift cannot be checked; pooled contrasts across runs "
                     "assume the model did not change between runs.")

    # Results by condition
    L.append("\n## Stated probability by condition\n")
    L.append("Mean of per-case means. pp = probability points (0–100)." + (" Cells pool all runs." if multi else "") + "\n")
    res_rows = []
    for cond in conditions:
        cm = [cells[(cid, cond)] for cid in case_ids if cells.get((cid, cond)) and cells[(cid, cond)]["n_valid"]]
        sds = [c["sd_pp"] for c in cm if c["sd_pp"] is not None]
        res_rows.append([cond, len(cm), fnum(statistics.fmean(c["mean_pp"] for c in cm)) if cm else "–",
                         fnum(statistics.fmean(c["mean_logodds"] for c in cm), 2) if cm else "–",
                         fnum(statistics.fmean(sds)) if sds else "–"])
    L.append(md_table(["Condition", "Cases", "Mean pp", "Mean log-odds", "Mean within-cell SD (pp)"], res_rows))

    # Contrasts
    L.append("\n## Pre-specified contrasts (per case, then across cases)\n")
    L.append("CI = 95% percentile bootstrap over cases. Sign test: two-sided exact, ties dropped. "
             "With 8 cases these are descriptive, not confirmatory.\n")
    crow = []
    for c in contrast_results:
        if "skipped" in c:
            crow.append([c["contrast"], "–", "skipped: " + c["skipped"], "", "", "", ""])
            continue
        pp, lo = c["pp"], c["logodds"]
        crow.append([c["contrast"], pp["n_cases"], fmt(pp["mean"]), f"[{fmt(pp['ci95_low'])}, {fmt(pp['ci95_high'])}]",
                     f"{fmt(lo['mean'], 2)} [{fmt(lo['ci95_low'], 2)}, {fmt(lo['ci95_high'], 2)}]",
                     f"{pp['n_pos']}/{pp['n_neg']}/{pp['n_zero']}",
                     fnum(pp["sign_test_p"], 3) if pp["sign_test_p"] is not None else "–"])
    L.append(md_table(["Contrast", "Cases", "Mean pp", "95% CI pp", "Mean log-odds [95% CI]", "+/−/0 cases", "Sign p"], crow))
    L.append("")
    for c in contrast_results:
        L.append(f"- **{c['contrast']}**: {c['description']}" + (f" Dropped cases (missing cells): {c['dropped_cases']}." if c.get("dropped_cases") else ""))

    # Manipulation checks
    L.append("\n## Manipulation check" + ("s" if len(GATES) > 1 else "") + "\n")
    for gname, sign, rule in GATES:
        g = next((c for c in contrast_results if c["contrast"] == gname and "skipped" not in c), None)
        if g is None:
            L.append(f"- {gname}: not available (its conditions were not run).")
            continue
        lo = g["logodds"]
        bound = lo["ci95_low"] if sign > 0 else lo["ci95_high"]
        passed = lo["mean"] is not None and bound is not None and sign * lo["mean"] > 0 and sign * bound > 0
        L.append(f"- {gname}: " + ("PASSED" if passed else "**NOT PASSED**") +
                 f". Mean {fmt(lo['mean'], 2)} log-odds, CI {'lower' if sign > 0 else 'upper'} bound {fmt(bound, 2)}. "
                 f"Rule (README): {rule}." + ("" if passed else " If it fails, the measure or the evidence is not working, "
                                               "so don't interpret the contrasts that depend on it."))

    # Per-case table
    L.append("\n## Per-case cell means (pp; n valid / planned)\n")
    pc_rows = []
    for cid in case_ids:
        row = [cid]
        for cond in conditions:
            c = cells.get((cid, cond))
            row.append("–" if not c or not c["n_valid"] else f"{c['mean_pp']:.0f} ({c['n_valid']}/{c['n_planned']})")
        pc_rows.append(row)
    L.append(md_table(["Case"] + conditions, pc_rows))
    base = [(cid, cells[(cid, "observation_only")]["mean_pp"]) for cid in case_ids
            if cells.get((cid, "observation_only")) and cells[(cid, "observation_only")]["n_valid"]]
    edge = [f"{cid} ({v:.0f})" for cid, v in base if v >= 90 or v <= 10]
    if edge:
        L.append(f"\nBaseline near floor or ceiling (≤10 or ≥90): {', '.join(edge)}. Their probability-point contrasts are compressed.")

    L.append("\n## Scope\n")
    L.append("Fictional cases, one model, prompt-only manipulation. This measures what a model states when given these "
             "notes. It does not measure real users' beliefs or any production memory system. Treat these numbers as "
             "pilot estimates for planning a larger study, not as evidence for a claim.")
    return "\n".join(L) + "\n", data


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("x", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    run_dirs = [Path(a) for a in argv]
    missing = [str(d) for d in run_dirs if not (d / "manifest.json").exists()]
    if missing:
        print(f"No manifest.json in: {missing}", file=sys.stderr)
        return 2
    try:
        report, data = analyse(run_dirs)
    except IncompatibleRuns as e:
        print("Refusing to combine these runs:", file=sys.stderr)
        for p in e.args[0]:
            print("  -", p, file=sys.stderr)
        return 3
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if len(run_dirs) == 1:
        out = run_dirs[0] / f"analysis-{stamp}"
    else:
        out = run_dirs[0].parent / "combined" / f"{stamp}_{len(run_dirs)}runs"
        out.parent.mkdir(exist_ok=True)
    out.mkdir()  # new folder every time; earlier analyses are kept
    if len(run_dirs) > 1:
        write_new = out / "sources.json"
        with write_new.open("x", encoding="utf-8") as fh:
            json.dump({"run_ids": data["run_ids"], "run_dirs": [str(d) for d in run_dirs]}, fh, indent=2)
    with (out / "report.md").open("x", encoding="utf-8") as fh:
        fh.write(report)
    write_csv(out / "parsed_calls.csv", data["rows"])
    write_csv(out / "cells.csv", [{k: v for k, v in c.items()} for c in data["cells"].values()])
    write_csv(out / "contrasts_per_case.csv", data["per_case_contrasts"])
    flat = []
    for c in data["contrasts"]:
        if "skipped" in c:
            flat.append({"contrast": c["contrast"], "skipped": c["skipped"]})
            continue
        for scale in ("pp", "logodds"):
            flat.append({"contrast": c["contrast"], "scale": scale, **c[scale], "dropped_cases": c["dropped_cases"]})
    write_csv(out / "contrasts_summary.csv", flat)
    if data["drift"]:
        write_csv(out / "drift.csv", [{"condition": d["condition"], "reference_run": d["reference_run"],
                                       "later_run": d["later_run"], "scale": sc, **d[sc]}
                                      for d in data["drift"] for sc in ("pp", "logodds")])
    print(report)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
