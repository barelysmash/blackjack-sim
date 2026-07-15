"""Post-simulation statistics.

Everything here consumes the raw ``RoundRecord`` list from the engine, so
adding a new metric never touches (or slows) the simulation hot path.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

from .engine import RoundRecord


def summarize(records: List[RoundRecord], min_bet: float) -> Dict[str, float]:
    n = len(records)
    if n == 0:
        return {"rounds": 0}
    nets = [r.net for r in records]
    bets = [r.bet for r in records]
    mean = sum(nets) / n
    var = sum((x - mean) ** 2 for x in nets) / max(n - 1, 1)
    sd = math.sqrt(var)
    total_wagered = sum(bets)
    return {
        "rounds": n,
        "total_net": sum(nets),
        "ev_per_round": mean,
        "ev_per_100_rounds": mean * 100,
        "edge_pct_of_action": 100.0 * sum(nets) / total_wagered if total_wagered else 0.0,
        "sd_per_round": sd,
        "avg_bet": total_wagered / n,
        "final_bankroll": records[-1].bankroll,
        "se_ev": sd / math.sqrt(n),  # standard error of the EV estimate
    }


def by_true_count(records: List[RoundRecord]) -> Dict[int, Dict[str, float]]:
    """EV per unit wagered, bucketed by floored true count at bet time.

    This table is the empirical basis for a bet ramp: bet in proportion
    to the per-unit edge in each bucket (Kelly).
    """
    buckets: Dict[int, List[RoundRecord]] = defaultdict(list)
    for r in records:
        tc = max(-5, min(6, math.floor(r.true_count)))
        buckets[tc].append(r)
    out: Dict[int, Dict[str, float]] = {}
    for tc in sorted(buckets):
        rs = buckets[tc]
        wagered = sum(r.bet for r in rs)
        net = sum(r.net for r in rs)
        edge = net / wagered if wagered else 0.0
        out[tc] = {
            "rounds": len(rs),
            "edge_per_unit": edge,
            "edge_pct": 100.0 * edge,
        }
    return out


def risk_of_ruin(ev_per_round: float, sd_per_round: float, bankroll: float) -> float:
    """Exponential (diffusion) approximation of lifetime risk of ruin.

    RoR = exp(-2 * EV * bankroll / variance). Only meaningful when EV > 0.
    """
    if ev_per_round <= 0:
        return 1.0
    var = sd_per_round ** 2
    return math.exp(-2.0 * ev_per_round * bankroll / var)


def kelly_ramp(tc_table: Dict[int, Dict[str, float]], max_units: float = 12.0) -> Dict[int, float]:
    """Derive a bet ramp (units per true count) from measured per-TC edges.

    units(tc) proportional to edge(tc) / variance, normalized so the first
    positive-edge bucket bets 1 unit. Negative-edge buckets bet the minimum.
    """
    variance = 1.32
    positive = {tc: v["edge_per_unit"] for tc, v in tc_table.items() if v["edge_per_unit"] > 0}
    if not positive:
        return {tc: 1.0 for tc in tc_table}
    base = min(e for e in positive.values() if e > 0)
    ramp = {}
    for tc, v in tc_table.items():
        e = v["edge_per_unit"]
        ramp[tc] = 1.0 if e <= 0 else min(round(e / base, 1), max_units)
    _ = variance  # kept explicit for readers extending to true Kelly dollars
    return ramp
