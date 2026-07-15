"""Monte Carlo learning.

Two learners, matching the two halves of the MIT playbook:

1. ``MCStrategyLearner`` — first-visit Monte Carlo control with an
   epsilon-greedy policy. Learns the playing strategy *from scratch* by
   sampling episodes and averaging returns per (state, action). With
   enough episodes the greedy policy converges to book basic strategy —
   a nice end-to-end validation that the engine's payouts are correct.

   State: (kind, key, dealer_up) where kind is 'hard'/'soft'/'pair'.
   Actions: H, S, D, P (legality-filtered per state).

2. ``learn_bet_ramp`` — plays flat-bet basic strategy for N shoes,
   measures the realized edge per true-count bucket, and derives a
   Kelly-proportional bet ramp from the measurements. This reproduces,
   from data, the count->bet mapping the MIT teams carried as a card.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

from .betting import FlatBet
from .cards import build_shoe, hand_total, is_blackjack
from .engine import Simulator
from .rules import Rules
from .stats import by_true_count, kelly_ramp
from .strategy import BasicStrategy

State = Tuple[str, int, int]          # (kind, key, dealer_up)
SA = Tuple[State, str]                # (state, action)


def _state(cards: List[int], dealer_up: int, can_split: bool) -> State:
    total, soft = hand_total(cards)
    if can_split and len(cards) == 2 and cards[0] == cards[1]:
        return ("pair", cards[0], dealer_up)
    if soft:
        return ("soft", total, dealer_up)
    return ("hard", total, dealer_up)


class MCStrategyLearner:
    """First-visit MC control, epsilon-greedy, tabular Q."""

    def __init__(self, rules: Rules | None = None, seed: int | None = None,
                 epsilon: float = 0.25):
        self.rules = rules or Rules()
        self.rng = random.Random(seed)
        self.epsilon = epsilon
        self.q_sum: Dict[SA, float] = defaultdict(float)
        self.q_n: Dict[SA, int] = defaultdict(int)
        self.episodes = 0

    # -- policy --------------------------------------------------------
    def _legal(self, cards: List[int], can_double: bool, can_split: bool) -> List[str]:
        acts = ["H", "S"]
        if can_double:
            acts.append("D")
        if can_split:
            acts.append("P")
        return acts

    def q(self, s: State, a: str) -> float:
        n = self.q_n[(s, a)]
        return self.q_sum[(s, a)] / n if n else 0.0

    def _choose(self, s: State, legal: List[str]) -> str:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal)
        return max(legal, key=lambda a: (self.q(s, a), self.q_n[(s, a)]))

    # -- episodes ------------------------------------------------------
    def train(self, episodes: int) -> None:
        r = self.rules
        for _ in range(episodes):
            # Fresh shoe slice per episode: near-iid deals, no count signal,
            # so the learner targets pure basic strategy.
            shoe = build_shoe(r.num_decks, self.rng)
            trajectory: List[SA] = []
            net = self._episode(shoe, trajectory)
            seen = set()
            for sa in trajectory:            # first-visit updates
                if sa in seen:
                    continue
                seen.add(sa)
                self.q_sum[sa] += net
                self.q_n[sa] += 1
            self.episodes += 1
            # Anneal exploration over the run.
            if self.episodes % 100_000 == 0:
                self.epsilon = max(0.05, self.epsilon * 0.9)

    def _episode(self, shoe: List[int], trajectory: List[SA]) -> float:
        r = self.rules
        player = [shoe.pop(), shoe.pop()]
        up = shoe.pop()
        hole = shoe.pop()
        dealer = [up, hole]
        if r.dealer_peeks and up >= 10 and is_blackjack(dealer):
            return 0.0 if is_blackjack(player) else -1.0
        if is_blackjack(player):
            return r.blackjack_payout

        hands: List[Tuple[List[int], float, bool]] = [(player, 1.0, False)]
        net_bets: List[Tuple[List[int], float]] = []
        i = 0
        while i < len(hands):
            cards, bet, from_split_ace = hands[i]
            if from_split_ace and not r.hit_split_aces:
                net_bets.append((cards, bet))
                i += 1
                continue
            while True:
                total, _ = hand_total(cards)
                if total >= 21:
                    break
                two = len(cards) == 2
                can_double = two and (len(hands) == 1 or r.double_after_split)
                can_split = two and cards[0] == cards[1] and len(hands) < r.max_hands
                s = _state(cards, up, can_split)
                a = self._choose(s, self._legal(cards, can_double, can_split))
                trajectory.append((s, a))
                if a == "H":
                    cards.append(shoe.pop())
                elif a == "S":
                    break
                elif a == "D":
                    bet *= 2
                    cards.append(shoe.pop())
                    break
                elif a == "P":
                    split_ace = cards[0] == 11
                    moved = cards.pop()
                    cards.append(shoe.pop())
                    hands.append(([moved, shoe.pop()], 1.0, split_ace))
                    hands[i] = (cards, bet, split_ace)
                    if split_ace and not r.hit_split_aces:
                        break
            net_bets.append((cards, bet))
            i += 1

        while True:
            total, soft = hand_total(dealer)
            if total > 21 or total > 17:
                break
            if total == 17 and (r.dealer_stands_soft_17 or not soft):
                break
            dealer.append(shoe.pop())
        dealer_total, _ = hand_total(dealer)

        net = 0.0
        for cards, bet in net_bets:
            total, _ = hand_total(cards)
            if total > 21:
                net -= bet
            elif dealer_total > 21 or total > dealer_total:
                net += bet
            elif total < dealer_total:
                net -= bet
        return net

    # -- output --------------------------------------------------------
    def greedy_policy(self) -> Dict[State, str]:
        policy: Dict[State, str] = {}
        states = {s for (s, _a) in self.q_n}
        for s in states:
            acts = [a for a in ("H", "S", "D", "P") if self.q_n[(s, a)] > 0]
            if acts:
                policy[s] = max(acts, key=lambda a: self.q(s, a))
        return policy

    def compare_to_book(self) -> Tuple[int, int, List[str]]:
        """Return (matches, total, diff_lines) vs. book basic strategy."""
        book = BasicStrategy()
        policy = self.greedy_policy()
        matches, total, diffs = 0, 0, []
        for (kind, key, up), learned in sorted(policy.items()):
            if kind == "hard":
                cards = [key - 10, 10] if key > 11 else [key - 2, 2]
            elif kind == "soft":
                cards = [11, key - 11]
            else:
                cards = [key, key]
            b = book.decide(cards, up, True, kind == "pair", 0.0)
            b = {"Ds": "D"}.get(b, b)
            total += 1
            if b == learned:
                matches += 1
            else:
                n = sum(self.q_n[((kind, key, up), a)] for a in "HSDP")
                diffs.append(
                    f"{kind:5s} {key:2d} vs {up:2d}: learned {learned} "
                    f"(book {b}, n={n})"
                )
        return matches, total, diffs


def learn_bet_ramp(
    shoes: int = 2_000,
    rules: Rules | None = None,
    seed: int | None = None,
    max_units: float = 12.0,
):
    """Measure edge per true count with flat bets, derive a Kelly ramp."""
    rules = rules or Rules()
    sim = Simulator(rules, BasicStrategy(use_deviations=False), FlatBet(), seed)
    res = sim.run(shoes, bankroll=1e9, min_bet=1.0, stop_on_ruin=False)
    table = by_true_count(res.records)
    return table, kelly_ramp(table, max_units=max_units)
