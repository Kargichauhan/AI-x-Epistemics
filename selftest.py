#!/usr/bin/env python3
"""End-to-end self-test of pilot.py and analyze.py. No network, no real key.

    .venv/bin/python selftest.py [--out-dir validation_runs]

Uses mock_server.py, whose responses are SYNTHETIC with known injected effects.
Checks:
  1. design checks pass on the real cases and catch deliberately broken ones
  2. response parser on known inputs
  3. dry run writes prompts/plan and makes no calls
  4. an existing run folder is never overwritten
  5. guards: missing credentials, --max-calls
  6. full mock run (Anthropic SDK path): every planned call recorded, each
     injected fault visible with the right label, requests on the wire match
     the recorded prompts, the dummy key is never written, analysis recovers
     the injected effects
  7. mock run through the OpenAI SDK path
  8. Ctrl-C mid-run -> status 'interrupted', unattempted calls reported
  9. hard kill mid-run -> missing run_end.json reported
 10. Adaption: model listing command, Responses-API run (store/agentic_memory
     false on the wire, failed-status responses recorded), cost accounting,
     price and --effort guards, --case-ids smoke run
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402
import pilot  # noqa: E402

PY = sys.executable
DUMMY_KEY = "LOCAL-MOCK-KEY-not-a-real-secret-7f3a"
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  -- {detail}" if detail and not cond else ""))
    return cond


def start_mock(log, *extra):
    proc = subprocess.Popen([PY, str(HERE / "mock_server.py"), "--log", str(log), *extra],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = proc.stdout.readline()
    return proc, int(line.split()[1])


def env_with(**kv):
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL",
                        "ADAPTION_API_KEY", "ADAPTION_BASE_URL", "ADAPTION_CUSTOM_HEADERS")}
    env.update(kv)
    return env


def run(cmd, env=None, timeout=300):
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)


def latest_analysis(run_dir: Path) -> Path:
    return sorted(run_dir.glob("analysis-*"))[-1]


def outcomes(run_dir: Path):
    _, data = analyze.analyse(run_dir)
    from collections import Counter
    return Counter(r["outcome"] for r in data["rows"]), data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "validation_runs")
    a = ap.parse_args()
    root = a.out_dir / ("selftest-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    root.mkdir(parents=True)
    print(f"Self-test folder: {root}\n")

    # 1. design checks
    _, cases = pilot.load_cases(HERE / "pilot_cases.json")
    check("design checks pass on pilot_cases.json", pilot.validate_design(cases) == [], pilot.validate_design(cases))
    bad = copy.deepcopy(cases)
    bad[0]["new_evidence"] = "This definitely shows that " + bad[0]["hypothesis"] + "."
    probs = pilot.validate_design(bad)
    check("design checks catch evidence that repeats the hypothesis / uses evaluative words",
          any("repeats the hypothesis" in p for p in probs) and any("evaluative" in p for p in probs), probs)
    bad = copy.deepcopy(cases)
    bad[1]["hypothesis"] = "The coworker is busy."
    check("design checks catch a malformed hypothesis", any("lower-case" in p for p in pilot.validate_design(bad)))
    saved = pilot.SLOT_A["attributed"]
    pilot.SLOT_A["attributed"] = "According to the AI assistant, which was speculating, {hypothesis}."
    check("design checks catch an attributed line that is not a minimal pair",
          any("differ by more than the source prefix" in p for p in pilot.validate_design(cases)))
    pilot.SLOT_A["attributed"] = saved
    lt = pilot.length_table(cases)
    check("attributed vs unlabelled: same word count in every case",
          all(d["attributed_minus_unlabelled_words"] == 0 for d in lt["attributed_vs_unlabelled"]))

    # 2. parser
    P = analyze.parse_response
    cases_p = [
        ('{"probability": 40, "reason": "x"}', "end_turn", "ok", 40.0),
        ('```json\n{"probability": 40, "reason": "x"}\n```', "end_turn", "ok", 40.0),
        ('Sure! {"probability": "35%", "reason": "x"}', "stop", "ok", 35.0),
        ('{"probability": 0.35}', "end_turn", "ambiguous_scale", 0.35),
        ('{"probability": 120}', "end_turn", "out_of_range", 120.0),
        ('{"probability": "high"}', "end_turn", "not_a_number", None),
        ('{"probability": 4', "max_tokens", "truncated", None),
        ("About 60%.", "end_turn", "no_json", None),
        ("", "refusal", "refusal", None),
        (None, "content_filter", "refusal", None),
        ("", "end_turn", "empty", None),
        ("", "max_output_tokens", "truncated", None),  # seen live: all output budget spent on reasoning
        ('{"probability": 30} or {"probability": 50}', "end_turn", "conflicting_json", None),
        ('{"probability": 0, "reason": "x"}', "end_turn", "ok", 0.0),
        ('{"probability": 100, "reason": "x"}', "end_turn", "ok", 100.0),
    ]
    bad_parse = [(t, s, P(t, s)) for t, s, st, pv in cases_p if (P(t, s)["status"], P(t, s)["probability"]) != (st, pv)]
    check(f"parser handles {len(cases_p)} known inputs", not bad_parse, bad_parse)

    # 3. dry run
    r = run([PY, str(HERE / "pilot.py"), "--dry-run", "--out-dir", str(root), "--run-id", "dry"], env=env_with())
    d = root / "dry"
    end = json.loads((d / "run_end.json").read_text()) if (d / "run_end.json").exists() else {}
    n_prompts = sum(1 for _ in (d / "prompts.jsonl").open()) if (d / "prompts.jsonl").exists() else 0
    n_expected = sum(len(pilot.supported_conditions(c)) for c in cases)  # 8 x 6 with the v2 cases file
    check(f"dry run exits 0, writes manifest + {n_expected} prompts, no calls.jsonl",
          r.returncode == 0 and (d / "manifest.json").exists() and n_prompts == n_expected and not (d / "calls.jsonl").exists()
          and end.get("status") == "dry_run", r.stderr[-500:])
    plan_conds = json.loads((d / "manifest.json").read_text())["settings"]["conditions"] if (d / "manifest.json").exists() else []
    check("default condition set is still the five main-pilot conditions", plan_conds == pilot.MAIN_FIVE, plan_conds)
    _, v3 = pilot.load_cases(HERE / "pilot_cases_v3.json")
    check("v3 cases pass design checks, and their v2 fields equal pilot_cases.json exactly",
          pilot.validate_design(v3) == [] and all({k: b[k] for k in a} == a for a, b in zip(cases, v3)))

    # 4. overwrite guard
    before = {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}
    r = run([PY, str(HERE / "pilot.py"), "--dry-run", "--out-dir", str(root), "--run-id", "dry"], env=env_with())
    after = {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}
    check("re-using a run id is refused and leaves files byte-identical",
          r.returncode == 4 and "Refusing to overwrite" in r.stderr and before == after, r.stderr[-300:])

    # 5. guards
    r = run([PY, str(HERE / "pilot.py"), "--provider", "openai", "--model", "x", "--out-dir", str(root), "--run-id", "nocreds"], env=env_with())
    check("missing OPENAI_API_KEY: exit 2, no run folder created",
          r.returncode == 2 and "OPENAI_API_KEY" in r.stderr and not (root / "nocreds").exists(), r.stderr[-300:])
    r = run([PY, str(HERE / "pilot.py"), "--provider", "openai", "--model", "x", "--repeats", "10", "--out-dir", str(root),
             "--run-id", "toomany"], env=env_with(OPENAI_API_KEY=DUMMY_KEY))
    check("--max-calls guard refuses a 400-call run", r.returncode == 2 and "max-calls" in r.stderr and not (root / "toomany").exists())

    expected_pp = {"carryover_unlabelled": 12, "carryover_attributed": 4, "attribution_reduction": 8,
                   "evidence_effect": 25, "evidence_effect_under_attribution": 25, "evidence_preservation": 0}
    expected_faults = {"api_error": 1, "no_json": 1, "truncated": 1, "out_of_range": 1, "refusal": 1, "ambiguous_scale": 1}

    # 6 + 7 + 10. full mock runs on all three SDK paths
    wire_keys = {
        "anthropic": lambda b: (b.get("system"), json.dumps(b.get("messages")), b.get("model"), b.get("max_tokens")),
        "openai": lambda b: (json.dumps(b.get("messages")), b.get("model"), b.get("max_completion_tokens")),
        "adaption": lambda b: (b.get("instructions"), b.get("input"), b.get("model"), b.get("max_output_tokens"),
                               b.get("store"), b.get("agentic_memory")),
    }
    PRICE_ARGS = ["--price-input", "1.5", "--price-output", "4"]
    for provider, repeats, base_suffix, keyvar, extra in (
            ("anthropic", 3, "", "ANTHROPIC_API_KEY", []),
            ("openai", 1, "/v1", "OPENAI_API_KEY", []),
            ("adaption", 1, "", "ADAPTION_API_KEY", PRICE_ARGS)):
        log = root / f"mock_{provider}_requests.jsonl"
        srv, port = start_mock(log)
        try:
            rid = f"mock_{provider}"
            r = run([PY, str(HERE / "pilot.py"), "--provider", provider, "--model", "mock-model", "--repeats", str(repeats),
                     "--base-url", f"http://127.0.0.1:{port}{base_suffix}", "--out-dir", str(root), "--run-id", rid, *extra],
                    env=env_with(**{keyvar: DUMMY_KEY}))
        finally:
            srv.terminate()
            srv.wait()
        exp_faults = dict(expected_faults, **({"response_failed": 1} if provider == "adaption" else {}))
        rd = root / rid
        n_plan = 40 * repeats
        calls = [json.loads(l) for l in (rd / "calls.jsonl").open()] if (rd / "calls.jsonl").exists() else []
        check(f"[{provider}] run completed and recorded all {n_plan} planned calls",
              r.returncode == 0 and len(calls) == n_plan and json.loads((rd / "run_end.json").read_text())["status"] == "completed",
              (r.stdout[-400:] + r.stderr[-400:]))
        oc, data = outcomes(rd)
        got_faults = {k: oc.get(k, 0) for k in exp_faults}
        check(f"[{provider}] every injected fault is visible with the right label", got_faults == exp_faults
              and oc.get("ok", 0) == n_plan - sum(exp_faults.values()), dict(oc))
        check(f"[{provider}] the HTTP 500 was retried by the SDK and recorded as such",
              sum((c.get("retries_taken") or 0) > 0 for c in calls) == 1)
        check(f"[{provider}] fenced JSON parsed and flagged", any("code_fence" in (row.get("parse_flags") or "") for row in data["rows"]))
        check(f"[{provider}] every call has raw response or error, model ids and timestamps",
              all((c["status"] == "ok" and c.get("response_raw") is not None and c.get("model_returned")) or
                  (c["status"] != "ok" and c.get("error")) for c in calls)
              and all(c.get("model_requested") == "mock-model" and c.get("started_at") and c.get("finished_at") for c in calls))
        # wire fidelity: what the server received equals what was recorded
        recv = [json.loads(l) for l in log.open()]
        key = wire_keys[provider]
        wire = {key(x["body"]) for x in recv}
        check(f"[{provider}] recorded request of every call matches a request seen on the wire",
              all(key(c["request"]) in wire for c in calls))
        if provider == "adaption":
            check("[adaption] every request sent store=false and agentic_memory=false (no carry-over between calls)",
                  recv and all(x["body"].get("store") is False and x["body"].get("agentic_memory") is False for x in recv))
            check("[adaption] notes sent as `instructions`, question as `input`",
                  all(x["body"].get("instructions", "").startswith("You are a helpful assistant.") and
                      x["body"].get("input", "").startswith("Picking up where we left off") for x in recv))
            end = json.loads((rd / "run_end.json").read_text())
            recomputed = sum(analyze.estimate_credits(analyze.normalise_usage(c.get("usage")),
                                                      {"input": 1.5, "cached_input": 1.5, "output": 4.0}) or 0 for c in calls)
            check("[adaption] run_end credit total equals recomputation from raw usage",
                  end.get("est_credits") is not None and abs(end["est_credits"] - recomputed) <= 1e-6 and recomputed > 0,  # stored rounded to 6 dp
                  (end.get("est_credits"), recomputed))
        # analysis recovers injected effects
        r2 = run([PY, str(HERE / "analyze.py"), str(rd)])
        rep = (latest_analysis(rd) / "report.md").read_text() if r2.returncode == 0 else ""
        got = {c["contrast"]: c["pp"]["mean"] for c in data["contrasts"] if "pp" in c}
        off = {k: round(got.get(k, float("nan")) - v, 2) for k, v in expected_pp.items()}
        check(f"[{provider}] analysis recovers injected effects within 1.5 pp", all(abs(x) <= 1.5 for x in off.values()), off)
        n_unusable = sum(exp_faults.values())
        check(f"[{provider}] report carries the SYNTHETIC banner and source line, and lists failures",
              "SYNTHETIC DATA" in rep and "SYNTHETIC local mock" in rep
              and f"Failures and unusable responses ({n_unusable})" in rep and "## Calls and cost" in rep, r2.stderr[-300:])
        leaked = [p for p in rd.rglob("*") if p.is_file() and DUMMY_KEY in p.read_text(errors="ignore")]
        check(f"[{provider}] dummy API key never written to any output file", not leaked, leaked)

    # 10. Adaption-specific commands and guards
    log = root / "mock_adaption_extra_requests.jsonl"
    srv, port = start_mock(log)
    try:
        base = f"http://127.0.0.1:{port}"
        env = env_with(ADAPTION_API_KEY=DUMMY_KEY)
        r = run([PY, str(HERE / "adaption_models.py"), "--base-url", base, "--out-dir", str(root / "model_listings")], env=env)
        saved = sorted((root / "model_listings").glob("*_adaption_models_SYNTHETIC.json"))
        check("[adaption] model listing: exit 0, SYNTHETIC label, mock ids shown, raw listing saved to a new file",
              r.returncode == 0 and "SYNTHETIC" in r.stdout and "mock-model-NOT-A-REAL-MODEL" in r.stdout
              and "2027-01-01" in r.stdout and len(saved) == 1
              and json.loads(saved[0].read_text())["listing"]["data"][0]["id"] == "mock-model-NOT-A-REAL-MODEL",
              r.stdout[-400:] + r.stderr[-400:])
        check("[adaption] model listing never prints or saves the key",
              DUMMY_KEY not in r.stdout + r.stderr and all(DUMMY_KEY not in p.read_text() for p in saved))
        r = run([PY, str(HERE / "adaption_models.py"), "--base-url", base, "--out-dir", str(root / "model_listings")], env=env_with())
        check("[adaption] model listing without ADAPTION_API_KEY: exit 2 with the hidden-input command",
              r.returncode == 2 and "read -rs" in r.stderr)
        r = run([PY, str(HERE / "pilot.py"), "--provider", "adaption", "--model", "unpriced-model", "--base-url", base,
                 "--out-dir", str(root), "--run-id", "noprice"], env=env)
        check("[adaption] unknown model price: refused before any call, no run folder",
              r.returncode == 2 and "No price known" in r.stderr and not (root / "noprice").exists(), r.stderr[-300:])
        r = run([PY, str(HERE / "pilot.py"), "--provider", "adaption", "--model", "mock-model", "--effort", "low",
                 "--base-url", base, "--out-dir", str(root), "--run-id", "effort", *PRICE_ARGS], env=env)
        check("[adaption] --effort refused (not a documented Adaption parameter)", r.returncode == 2 and not (root / "effort").exists())
        r = run([PY, str(HERE / "pilot.py"), "--provider", "adaption", "--model", "glm-5.2", "--repeats", "3",
                 "--max-tokens", "20000", "--base-url", base, "--out-dir", str(root), "--run-id", "toodear"], env=env)
        check("[adaption] --max-credits guard uses the documented price and refuses an expensive plan",
              r.returncode == 2 and "exceeds --max-credits" in r.stderr and not (root / "toodear").exists(), r.stderr[-300:])
        r = run([PY, str(HERE / "pilot.py"), "--provider", "adaption", "--model", "mock-model", "--case-ids", "garden_holes,cat_ears",
                 "--repeats", "1", "--base-url", base, "--out-dir", str(root), "--run-id", "smoke", *PRICE_ARGS], env=env)
        sm = [json.loads(l) for l in (root / "smoke" / "calls.jsonl").open()] if (root / "smoke" / "calls.jsonl").exists() else []
        check("[adaption] --case-ids smoke run covers all 5 conditions for the chosen cases",
              r.returncode == 0 and len(sm) == 10 and {c["condition"] for c in sm} == set(pilot.MAIN_FIVE)
              and {c["case_id"] for c in sm} == {"garden_holes", "cat_ears"}, r.stdout[-300:] + r.stderr[-300:])
        r2 = run([PY, str(HERE / "analyze.py"), str(root / "smoke")])
        check("[adaption] smoke analysis runs on the case subset", r2.returncode == 0 and "garden_holes" in r2.stdout
              and "electricity_bill" not in r2.stdout.split("## Per-case")[-1], r2.stderr[-300:])
    finally:
        srv.terminate()
        srv.wait()

    # 11. Extension conditions and combined analysis
    EXT = "observation_only,unlabelled_summary_plus_evidence,disconfirming_evidence"
    log = root / "mock_extension_requests.jsonl"
    srv, port = start_mock(log)
    try:
        env = env_with(ADAPTION_API_KEY=DUMMY_KEY)
        base = f"http://127.0.0.1:{port}"
        r = run([PY, str(HERE / "pilot.py"), "--provider", "adaption", "--model", "mock-model", "--conditions",
                 "disconfirming_evidence", "--base-url", base, "--out-dir", str(root), "--run-id", "ext_nov3", *PRICE_ARGS], env=env)
        check("[extension] disconfirming condition with the v2 cases file is refused before any call",
              r.returncode == 2 and "pilot_cases_v3.json" in r.stderr and not (root / "ext_nov3").exists(), r.stderr[-300:])
        r = run([PY, str(HERE / "pilot.py"), "--provider", "adaption", "--model", "mock-model", "--cases", str(HERE / "pilot_cases_v3.json"),
                 "--conditions", EXT, "--repeats", "1", "--base-url", base, "--out-dir", str(root), "--run-id", "mock_extension",
                 *PRICE_ARGS], env=env)
        check("[extension] mock run of the 3 extension conditions completes (24 calls)",
              r.returncode == 0 and sum(1 for _ in (root / "mock_extension" / "calls.jsonl").open()) == 24, r.stderr[-300:])
    finally:
        srv.terminate()
        srv.wait()
    r = run([PY, str(HERE / "analyze.py"), str(root / "mock_adaption"), str(root / "mock_extension")])
    combo = sorted((root / "combined").glob("*_2runs")) if (root / "combined").exists() else []
    rep = (combo[-1] / "report.md").read_text() if combo else ""
    check("[combined] analysis of main + extension runs exits 0 and writes results/combined/<time>_2runs/",
          r.returncode == 0 and combo and (combo[-1] / "sources.json").exists() and "Combined analysis" in rep, r.stderr[-400:])
    _, cdata = analyze.analyse([root / "mock_adaption", root / "mock_extension"])
    got = {c["contrast"]: c["pp"]["mean"] for c in cdata["contrasts"] if "pp" in c}
    exp = {"disconfirming_effect": -15, "interpretation_effect_with_evidence": 12, "attribution_reduction_with_evidence": 8,
           "carryover_unlabelled": 12, "evidence_effect": 25}
    off = {k: round(got.get(k, float("nan")) - v, 2) for k, v in exp.items()}
    check("[combined] recovers injected effects incl. disconfirming (-15) within 1.5 pp", all(abs(x) <= 1.5 for x in off.values()), off)
    drift = [d for d in cdata["drift"] if d["condition"] == "observation_only"]
    check("[combined] drift check on the repeated observation_only condition is about 0",
          drift and drift[0]["pp"]["mean"] is not None and abs(drift[0]["pp"]["mean"]) <= 1.5, drift)
    check("[combined] both manipulation checks reported, disconfirming gate passes on synthetic data",
          "evidence_effect: PASSED" in rep and "disconfirming_effect: PASSED" in rep and "SYNTHETIC DATA" in rep)
    # Perseverance conditions on the mock: unlabelled +12 / attributed +4 on top of disconfirming -15.
    srv, port = start_mock(root / "mock_perseverance_requests.jsonl")
    try:
        r = run([PY, str(HERE / "pilot.py"), "--provider", "adaption", "--model", "mock-model", "--cases", str(HERE / "pilot_cases_v3.json"),
                 "--conditions", "disconfirming_evidence,unlabelled_summary_plus_disconfirming,attributed_summary_plus_disconfirming",
                 "--repeats", "1", "--base-url", f"http://127.0.0.1:{port}", "--out-dir", str(root), "--run-id", "mock_perseverance",
                 *PRICE_ARGS], env=env_with(ADAPTION_API_KEY=DUMMY_KEY))
    finally:
        srv.terminate()
        srv.wait()
    _, pdata = analyze.analyse([root / "mock_perseverance"])
    got = {c["contrast"]: c["pp"]["mean"] for c in pdata["contrasts"] if "pp" in c}
    exp = {"perseverance_unlabelled": 12, "perseverance_attributed": 4, "attribution_reduction_with_disconfirming": 8}
    off = {k: round(got.get(k, float("nan")) - v, 2) for k, v in exp.items()}
    check("[perseverance] mock run completes and analysis recovers injected effects within 1.5 pp",
          r.returncode == 0 and all(abs(x) <= 1.5 for x in off.values()), (off, r.stderr[-300:]))
    _, p3 = analyze.analyse([root / "mock_extension", root / "mock_perseverance"])
    dd = [d for d in p3["drift"] if d["condition"] == "disconfirming_evidence"]
    check("[perseverance] combining with the extension run gives a drift check on disconfirming_evidence",
          dd and dd[0]["pp"]["mean"] is not None and abs(dd[0]["pp"]["mean"]) <= 1.5, dd)

    tampered = root / "mock_extension_tampered"
    shutil.copytree(root / "mock_extension", tampered)
    lines = (tampered / "prompts.jsonl").read_text().splitlines()
    lines = [l.replace("Thanks for your help so far.", "Thanks so much for your help so far.", 1)
             if json.loads(l)["condition"] == "observation_only" else l for l in lines]
    (tampered / "prompts.jsonl").write_text("\n".join(lines) + "\n")
    r = run([PY, str(HERE / "analyze.py"), str(root / "mock_adaption"), str(tampered)])
    check("[combined] refuses to pool runs whose shared prompts differ", r.returncode == 3 and "differs between" in r.stderr, r.stderr[-300:])

    # 8 + 9. interrupted and killed runs
    for how, sig in (("interrupted", signal.SIGINT), ("killed", signal.SIGKILL)):
        log = root / f"mock_{how}_requests.jsonl"
        srv, port = start_mock(log, "--slow-after", "5", "--slow-seconds", "60")
        rid = f"mock_{how}"
        p = subprocess.Popen([PY, str(HERE / "pilot.py"), "--provider", "anthropic", "--model", "mock-model", "--repeats", "1",
                              "--base-url", f"http://127.0.0.1:{port}", "--out-dir", str(root), "--run-id", rid],
                             env=env_with(ANTHROPIC_API_KEY=DUMMY_KEY), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        cf = root / rid / "calls.jsonl"
        t0 = time.time()
        while time.time() - t0 < 30 and not (cf.exists() and sum(1 for _ in cf.open()) >= 4):
            time.sleep(0.2)
        time.sleep(1.0)  # let the 6th request block on the slow server
        p.send_signal(sig)
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
        srv.terminate()
        srv.wait()
        rd = root / rid
        end_path = rd / "run_end.json"
        oc, _ = outcomes(rd)
        r2 = run([PY, str(HERE / "analyze.py"), str(rd)])
        rep = (latest_analysis(rd) / "report.md").read_text() if r2.returncode == 0 else ""
        if how == "interrupted":
            end = json.loads(end_path.read_text()) if end_path.exists() else {}
            check("Ctrl-C mid-run: run_end status 'interrupted' with the in-flight call named",
                  end.get("status") == "interrupted" and end.get("call_in_flight_when_stopped"), end)
            check("Ctrl-C mid-run: report flags incomplete run and counts unattempted calls",
                  "Run status: interrupted" in rep and oc.get("not_attempted", 0) == 40 - end.get("n_recorded", -1), dict(oc))
        else:
            check("hard kill mid-run: no run_end.json and the report says so",
                  not end_path.exists() and "run_end.json is missing" in rep and oc.get("not_attempted", 0) >= 30, dict(oc))

    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed. Outputs in {root}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
