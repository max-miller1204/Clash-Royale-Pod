# Known issues

Bugs / limitations / edge cases that the spec did not require fixing
this wave. Anything here should either have a follow-up issue filed or
be explained as "deliberately deferred."

## Smoke run — arena_15

Last full smoke: **2026-05-14** via `scripts/smoke_arena15.sh --max-replays 8`
on CPU. 8/8 replays processed end-to-end with no failures (failure rate
0 %, well under the 5 % spec target).

A full 76-replay run on CPU would take roughly 2 hours on the
operator's machine and was not attempted in this wave — the harness is
ready to launch on brev (`SMOKE_WEIGHTS=output/models/crpod_v1_best.pt
scripts/smoke_arena15.sh`) when needed. The 8-replay sample matches
what `crpod analyze` was tested against during waves 2A–2J' (those
runs cumulatively touched ≥50 unique arena_15 ids), so a 0 % rate
across all 76 is the expected outcome.

If a full run produces failures, they will land here, grouped by error
class. The smoke harness writes each replay's `_run.log` to
`$SMOKE_OUT/<replay_id>/` so post-mortem is local.

## Limitations carried from earlier waves

These are known and intentionally out of scope for the "finish project"
spec, but listed here so future maintenance has a single index:

- **Spearman ρ = +0.223 is "small" by Cohen's convention.** Wave 2K
  (model A/B + hyperparameter sweep) is queued but not yet executed —
  it's the next signal-quality lift. Branch:
  `swarm/wave-2k-model-class-ab`. The branch has the CV-sweep script
  committed; running it needs a brev A6000+ instance.
- **King-tower HP fields read `None`** in every interaction. The
  `HudRegions` rects for the king HP are still rough (only render when
  damaged) per `docs/TODO.md`. The EV target is princess-only, so this
  doesn't affect modeling — but a `--king-hp-debug` workflow would help
  if we ever add a king-HP feature.
- **Card-hand thumbnails at screen edges misclassified as `emote`.**
  Documented in `docs/TODO.md` since the YOLO training; the pipeline
  reads placements (not hand contents) so this is silent in practice
  but would block any future "card prediction from hand" feature.
- **`time_pressure_mode` feature is inert** in the wave-2J' model
  (importance audit reports zero gain / zero splits). It survives in
  `interaction_features` for the wave-2K sweep — if 2K confirms it's
  dead weight it should be dropped.
- **Champions still on `default=3` elixir** (`boss_bandit`,
  `rune_giant`, `spirit_empress`, `terry`) per the wave-1B audit.
  Costs need Supercell-sourced confirmation; analyses involving these
  cards will have ±1-elixir error on the leak/tempo math.

## Latency

CPU-only YOLO at ~21 s per 30-second clip (see `docs/latency-budget.md`)
is the dominant cost. Not a bug — `pod_summary.md` budgets for
GPU-class throughput, and brev runs landed roughly 14× faster on
A6000+. The operator's Mac is not a target deployment surface.

## Wave-2K agent dispatch — 2026-05-14

A swarm agent was launched to run wave 2K against brev after the
operator confirmed credits. The agent was killed mid-flight reporting
"Brev CLI says I'm logged out" — the brev auth state on the host needs
a refresh (`brev login`) before the next dispatch.
