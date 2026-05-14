"""Self-contained one-page HTML report — wave 4A.

`render_report(result, out_dir)` writes `report.html` next to
`summary.json` / `placements.png` / `tempo.png`. The report bundles:

- the summary fields (replay id, arena, interaction count, leak, blunder count)
- the top-N (default 5) blunders as a sortable-by-eye table
- the placement heatmap and tempo plot, embedded as base64 PNGs

No external CSS or JS. No network. Opens in Chrome / Safari offline.
"""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crpod.pipeline import AnalysisResult


_TOP_N_BLUNDERS = 5


def _img_tag(path: Path, alt: str) -> str:
    """Encode an on-disk PNG as a base64 `<img>` so the report opens
    standalone. Returns a placeholder div if the file is missing."""
    if not path.exists():
        return (
            f'<div class="img-missing">[{escape(alt)} unavailable — '
            f"check stderr for the viz-skipped warning]</div>"
        )
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{escape(alt)}" />'


def _blunder_rows(result: AnalysisResult) -> str:
    if not result.blunders:
        return (
            '<tr><td colspan="5" class="empty">'
            "No blunders flagged. Either the model thinks every play was within "
            "1σ of card-typical, or no EV model was supplied."
            "</td></tr>"
        )
    rows: list[str] = []
    for b in result.blunders[:_TOP_N_BLUNDERS]:
        rows.append(
            "<tr>"
            f"<td>{b.play_idx}</td>"
            f"<td><code>{escape(b.card)}</code></td>"
            f"<td>{b.ev_predicted:+.1f}</td>"
            f"<td>{b.per_card_median:+.1f}</td>"
            f"<td><strong>{b.sigma_below:.2f}σ</strong></td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_report(result: AnalysisResult, out_dir: Path) -> Path:
    """Render `out_dir/report.html` from a finished `AnalysisResult`.

    Expects `placements.png` / `tempo.png` to already exist in `out_dir`
    (written by `crpod.visualization.plots`). Missing image files degrade
    to in-line placeholder text rather than raising — same contract as
    the `[warn] viz skipped` path in `_cmd_analyze*`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.html"

    placements_img = _img_tag(out_dir / "placements.png", "placement heatmap")
    tempo_img = _img_tag(out_dir / "tempo.png", "elixir tempo")
    blunder_rows = _blunder_rows(result)

    replay_id = escape(result.replay.replay_id)
    arena = escape(result.replay.arena)
    n_plays = len(result.replay.plays)
    n_interactions = len(result.interactions)
    n_blunders = len(result.blunders)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>crpod report — {replay_id}</title>
<style>
  :root {{
    --bg: #fafafa;
    --fg: #1f2937;
    --muted: #6b7280;
    --accent: #2563eb;
    --bad: #dc2626;
    --rule: #e5e7eb;
    --code-bg: #f3f4f6;
  }}
  body {{
    background: var(--bg); color: var(--fg);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 32px;
    max-width: 960px; margin-left: auto; margin-right: auto;
  }}
  h1 {{ font-size: 26px; margin: 0; }}
  h2 {{ font-size: 18px; margin: 28px 0 10px; padding-bottom: 4px; border-bottom: 1px solid var(--rule); }}
  .subtitle {{ color: var(--muted); margin: 4px 0 24px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
  .card {{ background: white; border: 1px solid var(--rule); border-radius: 6px; padding: 12px 14px; }}
  .card .label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .card .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 14px; background: white; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--rule); }}
  th {{ color: var(--muted); font-weight: 600; background: #f9fafb; }}
  code {{ background: var(--code-bg); padding: 1px 5px; border-radius: 3px; font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 13px; }}
  .empty {{ color: var(--muted); font-style: italic; text-align: center; }}
  img {{ max-width: 100%; height: auto; border: 1px solid var(--rule); border-radius: 6px; background: white; }}
  .img-missing {{ padding: 24px; background: var(--code-bg); border: 1px dashed var(--rule); border-radius: 6px; color: var(--muted); text-align: center; }}
  .plots {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
  @media (min-width: 720px) {{ .plots {{ grid-template-columns: 1fr 1fr; }} }}
  footer {{ margin-top: 36px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--rule); padding-top: 12px; }}
</style>
</head>
<body>

<h1>Post-game report</h1>
<p class="subtitle">replay <code>{replay_id}</code> · arena <code>{arena}</code></p>

<h2>Summary</h2>
<div class="summary-grid">
  <div class="card"><div class="label">Plays</div><div class="value">{n_plays}</div></div>
  <div class="card"><div class="label">Interactions</div><div class="value">{n_interactions}</div></div>
  <div class="card"><div class="label">Friendly leak</div><div class="value">{result.friendly_leak:.1f}</div></div>
  <div class="card"><div class="label">Enemy leak</div><div class="value">{result.enemy_leak:.1f}</div></div>
  <div class="card"><div class="label">Blunders</div><div class="value" style="color: {("var(--bad)" if n_blunders else "var(--fg)")};">{n_blunders}</div></div>
</div>

<h2>Top blunders</h2>
<p class="subtitle">
  Plays whose predicted EV is more than 1σ below the training-fold median for that card.
  Sorted worst-first; up to {_TOP_N_BLUNDERS} shown.
</p>
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Card (anchor)</th>
      <th>Predicted EV</th>
      <th>Card median</th>
      <th>σ below</th>
    </tr>
  </thead>
  <tbody>
{blunder_rows}
  </tbody>
</table>

<h2>Visualizations</h2>
<div class="plots">
  <div>
    <h3 style="font-size: 14px; margin: 0 0 6px; color: var(--muted);">Placement heatmap</h3>
    {placements_img}
  </div>
  <div>
    <h3 style="font-size: 14px; margin: 0 0 6px; color: var(--muted);">Elixir tempo</h3>
    {tempo_img}
  </div>
</div>

<footer>
  Generated by <code>crpod</code>. Open this file in Chrome or Safari — no network required.
</footer>

</body>
</html>
"""
    out_path.write_text(html)
    return out_path
