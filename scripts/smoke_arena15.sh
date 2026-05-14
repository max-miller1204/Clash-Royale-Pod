#!/usr/bin/env bash
# Wave 4C — end-to-end smoke harness for the 76 arena_15 replays.
#
# Runs `crpod analyze` over each replay, captures failure count, and
# prints a summary at the end. Per the spec's verification clause, the
# target is `failure rate < 5%`. Remaining failures (after fixes) get
# documented in `docs/known-issues.md`.
#
# Usage:
#     scripts/smoke_arena15.sh                # CPU-only — slow, ~hours
#     scripts/smoke_arena15.sh --max-replays 5  # quick sanity run
#     SMOKE_OUT=/tmp/smoke scripts/smoke_arena15.sh
#
# Environment:
#     SMOKE_OUT     output dir (default: /tmp/crpod-smoke-arena15)
#     SMOKE_WEIGHTS YOLO weights path (default: output/models/crpod_v1_best.pt)
#     SMOKE_MODEL   EV model path (optional; gates blunders.json output)
#
# Exit codes:
#     0   smoke completed; failure rate < 5%
#     1   failure rate >= 5% — investigate, then update docs/known-issues.md
#     2   environment problem (missing weights, no replays listed)

set -uo pipefail

OUT="${SMOKE_OUT:-/tmp/crpod-smoke-arena15}"
WEIGHTS="${SMOKE_WEIGHTS:-output/models/crpod_v1_best.pt}"
MODEL="${SMOKE_MODEL:-}"

if [ ! -f "$WEIGHTS" ]; then
    echo "error: YOLO weights not found at $WEIGHTS" >&2
    echo "       set SMOKE_WEIGHTS or place weights at the default path" >&2
    exit 2
fi

mkdir -p "$OUT"
LOG="$OUT/_smoke.log"
: > "$LOG"

# Allow `--max-replays N` to short-circuit for development.
MAX_REPLAYS=""
if [ "${1:-}" = "--max-replays" ] && [ -n "${2:-}" ]; then
    MAX_REPLAYS="$2"
fi

echo "== listing arena_15 replays" | tee -a "$LOG"
REPLAYS=$(uv run crpod list-replays --arena arena_15 | awk '{print $2}')
TOTAL=$(echo "$REPLAYS" | wc -l | tr -d ' ')
if [ -z "$REPLAYS" ] || [ "$TOTAL" = "0" ]; then
    echo "error: no arena_15 replays returned by HF lister" >&2
    exit 2
fi
echo "found $TOTAL replays" | tee -a "$LOG"

if [ -n "$MAX_REPLAYS" ]; then
    REPLAYS=$(echo "$REPLAYS" | head -n "$MAX_REPLAYS")
    TOTAL="$MAX_REPLAYS"
    echo "(limited to first $MAX_REPLAYS)" | tee -a "$LOG"
fi

PASS=0
FAIL=0
declare -a FAILED_IDS=()

i=0
for rid in $REPLAYS; do
    i=$((i + 1))
    rid_out="$OUT/$rid"
    mkdir -p "$rid_out"

    CMD=(uv run crpod analyze arena_15 "$rid" --weights "$WEIGHTS" --out "$rid_out")
    if [ -n "$MODEL" ]; then
        CMD+=(--model "$MODEL")
    fi

    printf "[%3d/%d] %s ... " "$i" "$TOTAL" "$rid" | tee -a "$LOG"
    if "${CMD[@]}" >"$rid_out/_run.log" 2>&1; then
        echo "ok" | tee -a "$LOG"
        PASS=$((PASS + 1))
    else
        echo "FAIL" | tee -a "$LOG"
        FAIL=$((FAIL + 1))
        FAILED_IDS+=("$rid")
        # Tail the failure log to the smoke log so the operator can grep.
        echo "    -- last 10 lines of $rid_out/_run.log:" >>"$LOG"
        tail -n 10 "$rid_out/_run.log" >>"$LOG" 2>/dev/null || true
    fi
done

PCT=$(awk -v f="$FAIL" -v t="$TOTAL" 'BEGIN{ if (t==0) print 0; else printf "%.1f", 100.0 * f / t }')

echo "" | tee -a "$LOG"
echo "== smoke summary" | tee -a "$LOG"
echo "passed: $PASS / $TOTAL" | tee -a "$LOG"
echo "failed: $FAIL / $TOTAL  (${PCT}%)" | tee -a "$LOG"

if [ "$FAIL" -gt 0 ]; then
    echo "" | tee -a "$LOG"
    echo "failed replay ids:" | tee -a "$LOG"
    for r in "${FAILED_IDS[@]}"; do
        echo "  $r" | tee -a "$LOG"
    done
fi

# Spec verification target: < 5% failure rate.
THRESHOLD=$(awk -v p="$PCT" 'BEGIN{ print (p < 5.0) ? 0 : 1 }')
if [ "$THRESHOLD" = "0" ]; then
    echo "" | tee -a "$LOG"
    echo "✓ failure rate ${PCT}% < 5% — smoke passed" | tee -a "$LOG"
    exit 0
else
    echo "" | tee -a "$LOG"
    echo "✗ failure rate ${PCT}% ≥ 5% — investigate before declaring done" | tee -a "$LOG"
    exit 1
fi
