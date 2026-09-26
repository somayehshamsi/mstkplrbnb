#!/usr/bin/env bash
# Frozen final MSTKP benchmark -- stages in the required order.
#
#   bash run_final.sh laptop       laptop rehearsal: check + V + smoke + real-size pilot
#   bash run_final.sh check        machine / licence check (prints --jobs to use)
#   bash run_final.sh validate     experiment V (brute-force correctness, ~5 min)
#   bash run_final.sh smoke        full pipeline on small graphs (~30-60 min)
#   bash run_final.sh final        the whole final suite, stage by stage
#   bash run_final.sh optional     optional families (O1, O2, GLAZY)
#   bash run_final.sh confirm      confirmation family X on fresh instances
#   bash run_final.sh status       progress per family
#   bash run_final.sh report       collect + checks + analysis
#
# Long stages must survive a closed browser tab: start them detached,
#   setsid nohup bash run_final.sh final > final.out 2>&1 &          (Linux)
#   nohup caffeinate -i bash run_final.sh laptop > laptop.out 2>&1 &  (macOS)
# and follow with `tail -f final.out` or `bash run_final.sh status`.
# Every stage is idempotent: if the Jupyter server is culled or restarted,
# run the same command again -- finished jobs are skipped, jobs that were
# running are redone from scratch, nothing is overwritten.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
ROOT=${MSTKP_FINAL_ROOT:-$HOME/mstkp_final}
SMOKE=${MSTKP_SMOKE_ROOT:-$HOME/mstkp_smoke}
LAPTOP=${MSTKP_LAPTOP_ROOT:-$HOME/mstkp_laptop}   # never the server root
PILOT_FAMILIES=${PILOT_FAMILIES:-A,E,AGRB}
PILOT_TIME_LIMIT=${PILOT_TIME_LIMIT:-}            # testing only; unset = 600 s
SMOKE_ALLOW_STATUS=${SMOKE_ALLOW_STATUS:-}        # e.g. "error" if no Gurobi licence
JOBS=${JOBS:-auto}            # machine-check recommends 20 on 24 free cores
MEM=${MEM_BUDGET:-auto}       # machine-check recommends ~79 GB on 88 GB
GEN_JOBS=${GEN_JOBS:-8}

FS="$PY final_suite.py --root $ROOT"

stage() { echo; echo "=== $(date '+%F %T')  $*"; }

run_pipeline() {   # $1 = final_suite prefix, $2 = its root directory
  # Two shared pools instead of one stage per family: every run inside a pool
  # is independent, so the 20 parallel workers never sit idle waiting for the
  # slowest job of a small stage.  Pool 2 needs pool 1: ACUT uses the optima
  # found in A, and F uses the budgets that CAL + select-beta freeze.
  local S="$1" R="$2"
  stage "generate core instances";   $S generate --family core --jobs "$GEN_JOBS"
  stage "pool 1/2: A + E + G + AGRB + CAL"
  $S run --family A,E,G,AGRB,CAL --jobs "$JOBS" --mem-budget "$MEM"
  if [ ! -f "$R/frozen/F_beta.json" ]; then
    stage "select-beta (frozen rule)"; $S select-beta
  fi
  stage "generate F cells";          $S generate --family F --jobs "$GEN_JOBS"
  stage "pool 2/2: ACUT + F + B + BL + D + C + H + L + W"
  $S run --family ACUT,F,B,BL,D,C,H,L,W --jobs "$JOBS" --mem-budget "$MEM"
  stage "status";                    $S status --family core
}

smoke_stage() {   # $1 = smoke root
  local R="$1" S="$PY final_suite.py --profile smoke --root $1" rc=0
  run_pipeline "$S" "$R"
  stage "smoke: collect + checks + analysis"
  $S collect --family core || echo "!! collect reported integrity problems (see above)"
  $PY smoke_check.py --root "$R" --profile smoke --family core \
      ${SMOKE_ALLOW_STATUS:+--allow-status $SMOKE_ALLOW_STATUS} || rc=$?
  $PY analyze_final.py --root "$R" --profile smoke --family core --boot 2000
  if [ "$rc" = 0 ]; then echo "SMOKE TEST PASSED"; else echo "!! SMOKE CHECKS FAILED (see above)"; exit "$rc"; fi
}

case "${1:-}" in
  check)
    $FS machine-check ;;
  validate)
    mkdir -p "$ROOT"
    $PY validate_cuts.py --out "$ROOT/validation_report.json" ;;
  smoke)
    smoke_stage "$SMOKE" ;;
  laptop)
    # Rehearsal on a laptop.  Everything goes under $LAPTOP; the pilot uses
    # the real sizes but its own instances, so it never touches the final ones.
    P="$PY final_suite.py --profile pilot --root $LAPTOP/pilot"
    mkdir -p "$LAPTOP"
    stage "laptop: machine check";      $P machine-check
    stage "laptop: experiment V";       $PY validate_cuts.py --out "$LAPTOP/validation_report.json"
    stage "laptop: smoke pipeline";     smoke_stage "$LAPTOP/smoke"
    stage "laptop: real-size pilot ($PILOT_FAMILIES)"
    $P generate --family "$PILOT_FAMILIES" --jobs "$GEN_JOBS"
    $P run --family "$PILOT_FAMILIES" --jobs "$JOBS" --mem-budget "$MEM" \
        ${PILOT_TIME_LIMIT:+--time-limit $PILOT_TIME_LIMIT}
    $P collect --family "$PILOT_FAMILIES" || echo "!! collect reported integrity problems"
    rc=0
    $PY smoke_check.py --root "$LAPTOP/pilot" --profile pilot --family "$PILOT_FAMILIES" \
        ${SMOKE_ALLOW_STATUS:+--allow-status $SMOKE_ALLOW_STATUS} || rc=$?
    $PY analyze_final.py --root "$LAPTOP/pilot" --profile pilot --family "$PILOT_FAMILIES" --boot 2000
    stage "laptop: pilot run times (s) and peak memory (MB) per configuration"
    $PY - "$LAPTOP/pilot" <<'PYEOF'
import sys, glob, json, statistics as st
rows = {}
for p in glob.glob(sys.argv[1] + "/results/*/*/*.json"):
    r = json.load(open(p)); m = r["metrics"]
    rows.setdefault(r["key"]["config_id"], []).append((m.get("status"), m.get("wall_time") or 0,
                                                     r["run"].get("peak_rss_mb") or 0))
for c in sorted(rows):
    v = rows[c]
    print(f"  {c:22s} runs {len(v):2d}  solved {sum(s == 'optimal' for s, _, _ in v):2d}  "
          f"median time {st.median(t for _, t, _ in v):7.1f}  max memory {max(m for _, _, m in v):7.0f}")
PYEOF
    if [ "$rc" = 0 ]; then echo "LAPTOP REHEARSAL PASSED"; else echo "!! PILOT CHECKS FAILED"; exit "$rc"; fi ;;
  final)
    run_pipeline "$FS" "$ROOT" ;;
  optional)
    $FS generate --family O1,O2,GLAZY --jobs "$GEN_JOBS"
    $FS run --family O1,O2,GLAZY --jobs "$JOBS" --mem-budget "$MEM" ;;
  confirm)
    # Confirmation family X on fresh instances (after the core suite).
    $FS generate --family X --jobs "$GEN_JOBS"
    $FS run --family X --jobs "$JOBS" --mem-budget "$MEM"
    $FS collect --family X || echo "!! collect reported integrity problems (see above)"
    rc=0
    $PY smoke_check.py --root "$ROOT" --profile final --family X || rc=$?
    $PY analyze_final.py --root "$ROOT" --profile final --family X
    exit "$rc" ;;
  status)
    $FS status --family all ;;
  report)
    $FS collect --family "${2:-core}" || echo "!! collect reported integrity problems (see above)"
    rc=0
    $PY smoke_check.py --root "$ROOT" --profile final --family "${2:-core}" || rc=$?
    $PY analyze_final.py --root "$ROOT" --profile final --family "${2:-core}"
    exit "$rc" ;;
  *)
    sed -n '2,20p' "$0"; exit 1 ;;
esac
