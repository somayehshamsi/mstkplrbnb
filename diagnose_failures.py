#!/usr/bin/env python3
"""Read-only summary of runs that did not end optimal or by timeout.

    python3 diagnose_failures.py [RESULTS_ROOT]      (default ~/mstkp_final)

Safe to run while the benchmark is running: it only reads result files.
"""
import collections
import glob
import json
import os
import sys

root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/mstkp_final")
BAD = ("killed_memory", "memory", "crashed", "error", "killed_timeout")

rows = []
for path in glob.glob(os.path.join(root, "results", "*", "*", "*.json")):
    try:
        r = json.load(open(path))
    except Exception:
        continue
    status = r.get("metrics", {}).get("status")
    if status not in BAD:
        continue
    mon_path = path.replace(os.sep + "results" + os.sep, os.sep + "monitor" + os.sep)
    mon = json.load(open(mon_path)) if os.path.exists(mon_path) else {}
    err = [l for l in (r.get("error") or "").strip().splitlines() if l.strip()]
    rows.append({
        "status": status,
        "cell": r["key"]["cell_id"].split("_k")[0],
        "config": r["key"]["config_id"],
        "idx": r["key"]["idx"],
        "seconds": round(mon.get("elapsed") or r["metrics"].get("wall_time") or 0),
        "mb": round(mon.get("max_rss_mb") or 0),
        "limit_gb": r.get("run", {}).get("mem_limit_gb"),
        "first": err[0][:140] if err else "",
        "last": err[-1][:140] if err else "",
    })

print(f"results root: {root}")
print(f"{len(rows)} runs with status in {BAD}\n")

print("count  status          cell                           config")
for (st, cell, cfg), n in sorted(collections.Counter(
        (x["status"], x["cell"], x["config"]) for x in rows).items()):
    print(f"{n:5d}  {st:15s} {cell:30s} {cfg}")

print("\nwhen stopped (median over runs)   status / config: runs, seconds, peak MB, limit GB")
groups = collections.defaultdict(list)
for x in rows:
    groups[(x["status"], x["config"])].append(x)
for (st, cfg), g in sorted(groups.items()):
    s = sorted(x["seconds"] for x in g)
    m = sorted(x["mb"] for x in g)
    print(f"  {st:15s} {cfg:22s} runs {len(g):4d}  median {s[len(s) // 2]:6d} s  "
          f"{m[len(m) // 2]:7d} MB  limit {g[0]['limit_gb']}")

print("\ncrashed / error details")
for x in rows:
    if x["status"] in ("crashed", "error"):
        print(f"  {x['status']} {x['cell']} {x['config']} #{x['idx']:03d} after {x['seconds']} s, "
              f"{x['mb']} MB\n      first: {x['first']}\n      last : {x['last']}")
