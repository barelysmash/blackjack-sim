"""MIT-style team play: spotters, Big Players, one shared bankroll.

The solo counter's tell is bet variation correlated with a visible count.
The team structure breaks that correlation: spotters flat-bet the table
minimum and just keep the count; a Big Player is called in only when a
shoe goes hot, bets big from his first hand, and leaves when the edge
dies. The bet variation lives in *where the BP is standing*, not in his
chip stack.

Economics captured here:

* Spotters play every round at table minimum with plain basic strategy —
  a steady ~-0.5% cost charged against the team bankroll (the price of
  camouflage and table coverage).
* Each free BP is assigned to the highest-true-count table at or above
  the call-in threshold, plays deviations, sizes bets on the SHARED
  bankroll, and leaves when the count drops below the leave threshold or
  the shoe reshuffles.
* One tick = one round dealt at every table. BP hands ride the same
  shoes the spotters are counting.

Not modeled: heat/backoffs, table hopping delays, bet-size caps by pit,
tipping. Numbers are an upper bound on the clean math.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from .betting import KellyBet, SpreadBet
from .cards import build_shoe
from .counting import HiLoCounter
from .engine import RoundRecord, Simulator
from .rules import Rules
from .strategy import BasicStrategy


@dataclass
class _Table:
    shoe: List[int]
    counter: HiLoCounter
    bp: int = -1              # index of seated BP, -1 if none


@dataclass
class TeamResult:
    ticks: int = 0
    spotter_rounds: int = 0
    spotter_net: float = 0.0
    spotter_action: float = 0.0
    bp_rounds: int = 0
    bp_net: float = 0.0
    bp_action: float = 0.0
    call_ins: int = 0
    records: List[RoundRecord] = field(default_factory=list)  # per tick
    ruined: bool = False

    @property
    def team_net(self) -> float:
        return self.spotter_net + self.bp_net

    @property
    def bp_utilization(self) -> float:
        return self.bp_rounds / max(self.ticks, 1)


class TeamSimulator:
    def __init__(
        self,
        rules: Rules,
        num_tables: int = 4,
        num_bps: int = 1,
        call_in_tc: float = 2.0,
        leave_tc: float = 1.0,
        bp_betting: Optional[object] = None,
        seed: Optional[int] = None,
    ) -> None:
        if leave_tc > call_in_tc:
            raise ValueError("leave_tc must be <= call_in_tc")
        self.rules = rules
        self.num_tables = num_tables
        self.num_bps = num_bps
        self.call_in_tc = call_in_tc
        self.leave_tc = leave_tc
        self.rng = random.Random(seed)
        self.bp_betting = bp_betting or KellyBet(fraction=0.5, max_units=20)
        # Two players, one engine each, sharing shoes/counters per table:
        # spotters play plain basic strategy, BPs play with deviations.
        self._spotter = Simulator(rules, BasicStrategy(), None,
                                  seed=self.rng.randrange(2**32))
        self._bp = Simulator(rules, BasicStrategy(use_deviations=True), None,
                             seed=self.rng.randrange(2**32))

    def _fresh_table(self) -> _Table:
        return _Table(build_shoe(self.rules.num_decks, self.rng), HiLoCounter())

    def run(
        self,
        ticks: int,
        bankroll: float = 100_000.0,
        spotter_bet: float = 25.0,
        stop_on_ruin: bool = True,
    ) -> TeamResult:
        res = TeamResult()
        tables = [self._fresh_table() for _ in range(self.num_tables)]
        bp_seat: List[int] = [-1] * self.num_bps      # table index per BP

        for _ in range(ticks):
            if bankroll < spotter_bet * self.num_tables:
                res.ruined = True
                if stop_on_ruin:
                    break

            # Reshuffle depleted shoes (BP leaves on shuffle).
            for ti, t in enumerate(tables):
                if len(t.shoe) <= max(self.rules.reshuffle_at, 24):
                    t.shoe = build_shoe(self.rules.num_decks, self.rng)
                    t.counter.reset()
                    if t.bp >= 0:
                        bp_seat[t.bp] = -1
                        t.bp = -1

            # BPs leave cold tables, then free BPs take the hottest ones.
            tcs = [t.counter.true_count(len(t.shoe)) for t in tables]
            for bi in range(self.num_bps):
                ti = bp_seat[bi]
                if ti >= 0 and tcs[ti] < self.leave_tc:
                    tables[ti].bp = -1
                    bp_seat[bi] = -1
            open_hot = sorted(
                (ti for ti, t in enumerate(tables)
                 if t.bp < 0 and tcs[ti] >= self.call_in_tc),
                key=lambda ti: -tcs[ti],
            )
            for bi in range(self.num_bps):
                if bp_seat[bi] < 0 and open_hot:
                    ti = open_hot.pop(0)
                    bp_seat[bi] = ti
                    tables[ti].bp = bi
                    res.call_ins += 1

            # Deal one round everywhere.
            tick_net = 0.0
            best_tc = max(tcs)
            for ti, t in enumerate(tables):
                net = self._spotter._play_round(
                    t.shoe, t.counter, spotter_bet, bankroll)
                res.spotter_rounds += 1
                res.spotter_net += net
                res.spotter_action += spotter_bet
                tick_net += net
                if t.bp >= 0:
                    tc = t.counter.true_count(len(t.shoe))
                    bet = self.bp_betting.bet(tc, bankroll, spotter_bet)
                    if bet > 0:
                        bnet = self._bp._play_round(
                            t.shoe, t.counter, bet, bankroll)
                        res.bp_rounds += 1
                        res.bp_net += bnet
                        res.bp_action += bet
                        tick_net += bnet
            bankroll += tick_net
            res.ticks += 1
            res.records.append(
                RoundRecord(best_tc, spotter_bet, tick_net, bankroll))
        return res
