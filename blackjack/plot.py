"""Bankroll trajectory output — CSV for analysis, SVG for eyeballs.

Pure stdlib on purpose: the repo has zero dependencies and an SVG opens
in any browser. Long runs are downsampled to keep files small; the
min/max/final annotations are computed from the *full* series first, so
downsampling never hides the worst drawdown.
"""

from __future__ import annotations

from typing import List

from .engine import RoundRecord


def write_csv(records: List[RoundRecord], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("round,true_count,bet,net,bankroll\n")
        for i, r in enumerate(records, 1):
            f.write(f"{i},{r.true_count:.2f},{r.bet:.2f},{r.net:.2f},{r.bankroll:.2f}\n")


def write_svg(
    records: List[RoundRecord],
    path: str,
    title: str = "Bankroll trajectory",
    start_bankroll: float | None = None,
    max_points: int = 4000,
) -> None:
    if not records:
        raise ValueError("no rounds to plot")

    series = [r.bankroll for r in records]
    n = len(series)
    lo, hi = min(series), max(series)
    final = series[-1]
    start = start_bankroll if start_bankroll is not None else series[0]
    lo, hi = min(lo, start), max(hi, start)
    if hi == lo:
        hi = lo + 1.0

    stride = max(1, n // max_points)
    pts = series[::stride]
    if pts[-1] != final:
        pts.append(final)

    w, h = 900, 420
    ml, mr, mt, mb = 70, 20, 40, 40   # margins
    pw, ph = w - ml - mr, h - mt - mb

    def x(i: int) -> float:
        return ml + pw * i / max(len(pts) - 1, 1)

    def y(v: float) -> float:
        return mt + ph * (1.0 - (v - lo) / (hi - lo))

    poly = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(pts))
    color = "#2a7d2e" if final >= start else "#b3261e"

    gridlines = []
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        gy = y(v)
        gridlines.append(
            f'<line x1="{ml}" y1="{gy:.1f}" x2="{w - mr}" y2="{gy:.1f}" '
            f'stroke="#ddd" stroke-width="1"/>'
            f'<text x="{ml - 8}" y="{gy + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#666">${v:,.0f}</text>'
        )

    sy = y(start)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"
     viewBox="0 0 {w} {h}" font-family="sans-serif">
  <rect width="{w}" height="{h}" fill="white"/>
  <text x="{ml}" y="24" font-size="15" fill="#222">{title}</text>
  <text x="{w - mr}" y="24" font-size="12" fill="#666" text-anchor="end">
    {n:,} rounds &#183; start ${start:,.0f} &#183; final ${final:,.0f} &#183; min ${min(series):,.0f} &#183; max ${max(series):,.0f}</text>
  {''.join(gridlines)}
  <line x1="{ml}" y1="{sy:.1f}" x2="{w - mr}" y2="{sy:.1f}"
        stroke="#888" stroke-width="1" stroke-dasharray="5,4"/>
  <polyline points="{poly}" fill="none" stroke="{color}" stroke-width="1.5"/>
  <text x="{ml}" y="{h - 12}" font-size="11" fill="#666">round 1</text>
  <text x="{w - mr}" y="{h - 12}" font-size="11" fill="#666" text-anchor="end">round {n:,}</text>
</svg>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
