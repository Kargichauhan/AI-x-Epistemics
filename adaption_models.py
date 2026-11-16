#!/usr/bin/env python3
"""List the model IDs your Adaption API key can use.

    .venv/bin/python adaption_models.py          # hosted models (what pilot.py can call)
    .venv/bin/python adaption_models.py --all    # also harnesses and inference stacks

Calls GET /api/v1/models through the official `adaption` SDK. The key is read
from ADAPTION_API_KEY and is never printed or saved. The raw listing is saved to
a new file, results/model_listings/<UTC time>_adaption_models.json, and a table
is printed with the price and input modalities documented for each model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot import ADAPTION_DOC_MODALITIES, PRICE_SOURCE, PRICES, _to_plain, pkg_version  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="include harnesses and inference stacks")
    ap.add_argument("--base-url", default=None, help="override API base URL (testing)")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "model_listings")
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args(argv)

    if not os.environ.get("ADAPTION_API_KEY"):
        print("ADAPTION_API_KEY is not set in this terminal. Enter it without echo:\n"
              "  read -rs -p 'Adaption API key: ' ADAPTION_API_KEY && export ADAPTION_API_KEY && echo", file=sys.stderr)
        return 2
    try:
        import adaption
    except ImportError:
        print("The adaption SDK is not installed: .venv/bin/pip install -r requirements.txt", file=sys.stderr)
        return 2

    kw = {"timeout": a.timeout, "max_retries": 2}
    if a.base_url:
        kw["base_url"] = a.base_url
    client = adaption.Adaption(**kw)
    host = urlparse(str(client.base_url)).hostname or ""
    synthetic = host in {"127.0.0.1", "localhost", "::1"}
    params = {} if a.all else {"type": "hosted"}
    try:
        raw = client.models.with_raw_response.list(**params)
        listing = _to_plain(raw.parse())
    except adaption.AuthenticationError:
        print("The key was rejected (401). Check that ADAPTION_API_KEY holds a current key.", file=sys.stderr)
        return 1
    except adaption.APIStatusError as e:
        print(f"Model listing failed: HTTP {e.status_code}: {str(e)[:300]}", file=sys.stderr)
        return 1
    except adaption.APIConnectionError as e:
        print(f"Could not reach {client.base_url}: {e}", file=sys.stderr)
        return 1

    now = dt.datetime.now(dt.timezone.utc)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{now.strftime('%Y%m%dT%H%M%SZ')}_adaption_models{'_SYNTHETIC' if synthetic else ''}.json"
    record = {
        "retrieved_at": now.isoformat(timespec="seconds"),
        "synthetic_mock": synthetic,
        "base_url": str(client.base_url),
        "query": params,
        "request_id": raw.headers.get("x-request-id") or raw.headers.get("request-id"),
        "adaption_sdk": pkg_version("adaption"),
        "price_source": PRICE_SOURCE["adaption"],
        "listing": listing,
    }
    with out.open("x", encoding="utf-8") as fh:  # never overwrite an earlier listing
        json.dump(record, fh, indent=2, default=str)
        fh.write("\n")

    if synthetic:
        print("SYNTHETIC mock listing from a local test server; these are not your account's models.\n")
    models = listing.get("data") or []
    print(f"{len(models)} model(s) available to this key{' (hosted only)' if not a.all else ''}:\n")
    print(f"{'model id':28s} {'display name':24s} {'owner':10s} {'credits/1M in (cached) / out':30s} {'docs modalities':16s} shutdown")
    for m in models:
        mid = m.get("id", "?")
        p = PRICES.get(("adaption", mid))
        price = f"{p['input']} ({p['cached_input']}) / {p['output']}" if p else "not in docs table"
        print(f"{mid:28s} {str(m.get('display_name') or '–'):24s} {str(m.get('owned_by') or '–'):10s} "
              f"{price:30s} {ADAPTION_DOC_MODALITIES.get(mid, '–'):16s} {m.get('shutdown_date') or '–'}")
    print(f"\nSaved raw listing: {out}")
    print("Pass one of these ids to: .venv/bin/python pilot.py --provider adaption --model <id>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
