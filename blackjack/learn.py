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


# ---------------------------------------------------------------------------
# Count-aware MC control: learn the index plays (deviations) from scratch.
# ---------------------------------------------------------------------------

import math

from .counting import _HILO
from .strategy import DEVIATIONS

CState = Tuple[str, int, int, int]     # (kind, key, dealer_up, tc_bucket)
CSA = Tuple[CState, str]


class MCDeviationLearner:
    """MC control with the true count in the state.

    Same zero-book-input learning as ``MCStrategyLearner``, but each
    decision state carries a floored true-count bucket, and episodes are
    dealt from *count-stratified* shoes: partially depleted shoes
    constructed to a target Hi-Lo running count. Extreme counts are rare
    in nature (TC >= +4 is ~2-3% of rounds), so stratification is what
    makes the high-count cells learnable in reasonable time. Within each
    constructed shoe the count then evolves honestly as cards fall.

    With enough episodes the greedy action in a cell like hard 16 vs T
    flips from H to S as the bucket crosses the book index — the
    Illustrious 18 rediscovered from payouts alone.
    """

    def __init__(self, rules: Rules | None = None, seed: int | None = None,
                 epsilon: float = 0.30, tc_min: int = -3, tc_max: int = 5):
        self.rules = rules or Rules()
        self.rng = random.Random(seed)
        self.epsilon = epsilon
        self.tc_min, self.tc_max = tc_min, tc_max
        self.q_sum: Dict[CSA, float] = defaultdict(float)
        self.q_n: Dict[CSA, int] = defaultdict(int)
        self.episodes = 0

    # -- helpers -------------------------------------------------------
    def _bucket(self, tc: float) -> int:
        return max(self.tc_min, min(self.tc_max, math.floor(tc)))

    def q(self, s: CState, a: str) -> float:
        n = self.q_n[(s, a)]
        return self.q_sum[(s, a)] / n if n else 0.0

    def _choose(self, s: CState, legal: List[str]) -> str:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal)
        return max(legal, key=lambda a: (self.q(s, a), self.q_n[(s, a)]))

    def _stratified_shoe(self) -> Tuple[List[int], int]:
        """Build a depleted shoe whose removed cards sum to a target
        Hi-Lo running count. Returns (shoe, running_count)."""
        r = self.rules
        rng = self.rng
        total = 52 * r.num_decks
        decks_left = rng.uniform(1.0, r.num_decks)
        remaining = int(decks_left * 52)
        removed = total - remaining
        target_tc = rng.uniform(self.tc_min - 0.5, self.tc_max + 1.5)
        R = int(round(target_tc * decks_left))

        low_avail = 20 * r.num_decks    # 2-6
        high_avail = 20 * r.num_decks   # T,A
        neu_avail = 12 * r.num_decks    # 7-9
        R = max(-min(removed, high_avail), min(R, min(removed, low_avail)))
        n_neu = rng.randint(0, min(neu_avail, removed - abs(R)))
        if (removed - n_neu + R) % 2:
            n_neu += -1 if n_neu > 0 else 1
        n_low = (removed - n_neu + R) // 2
        n_high = removed - n_neu - n_low
        if n_low > low_avail or n_high > high_avail or n_low < 0 or n_high < 0:
            return self._stratified_shoe()   # infeasible corner; redraw

        counts = {v: 4 * r.num_decks for v in range(2, 12)}
        counts[10] = 16 * r.num_decks
        for n, pool in ((n_low, [2, 3, 4, 5, 6]),
                        (n_high, [10, 10, 10, 10, 11]),
                        (n_neu, [7, 8, 9])):
            for _ in range(n):
                while True:
                    v = rng.choice(pool)
                    if counts[v] > 0:
                        counts[v] -= 1
                        break
        shoe = [v for v, c in counts.items() for _ in range(c)]
        rng.shuffle(shoe)
        return shoe, n_low - n_high

    # -- training ------------------------------------------------------
    def train(self, episodes: int) -> None:
        done = 0
        while done < episodes:
            shoe, running = self._stratified_shoe()
            plays = 0
            while len(shoe) >= 30 and plays < 12 and done < episodes:
                trajectory: List[CSA] = []
                net, running = self._episode(shoe, running, trajectory)
                seen = set()
                for sa in trajectory:
                    if sa in seen:
                        continue
                    seen.add(sa)
                    self.q_sum[sa] += net
                    self.q_n[sa] += 1
                done += 1
                plays += 1
                self.episodes += 1
                if self.episodes % 500_000 == 0:
                    self.epsilon = max(0.05, self.epsilon * 0.9)

    def _episode(self, shoe: List[int], running: int,
                 trajectory: List[CSA]) -> Tuple[float, int]:
        r = self.rules

        def draw() -> int:
            nonlocal running
            c = shoe.pop()
            running += _HILO[c]
            return c

        def tc_bucket() -> int:
            return self._bucket(running / max(len(shoe) / 52.0, 0.5))

        player = [draw(), draw()]
        up = draw()
        hole = shoe.pop()               # hidden: not counted until reveal
        dealer = [up, hole]
        if r.dealer_peeks and up >= 10 and sum(dealer) == 21:
            running += _HILO[hole]
            return (0.0 if is_blackjack(player) else -1.0), running
        if is_blackjack(player):
            running += _HILO[hole]
            return r.blackjack_payout, running

        hands: List[Tuple[List[int], float, bool]] = [(player, 1.0, False)]
        settled: List[Tuple[List[int], float]] = []
        i = 0
        while i < len(hands):
            cards, bet, from_split_ace = hands[i]
            if from_split_ace and not r.hit_split_aces:
                settled.append((cards, bet))
                i += 1
                continue
            while True:
                total, _ = hand_total(cards)
                if total >= 21:
                    break
                two = len(cards) == 2
                can_double = two and (len(hands) == 1 or r.double_after_split)
                can_split = two and cards[0] == cards[1] and len(hands) < r.max_hands
                kind_state = _state(cards, up, can_split)
                s: CState = (*kind_state, tc_bucket())
                legal = ["H", "S"] + (["D"] if can_double else []) \
                        + (["P"] if can_split else [])
                a = self._choose(s, legal)
                trajectory.append((s, a))
                if a == "H":
                    cards.append(draw())
                elif a == "S":
                    break
                elif a == "D":
                    bet *= 2
                    cards.append(draw())
                    break
                elif a == "P":
                    split_ace = cards[0] == 11
                    moved = cards.pop()
                    cards.append(draw())
                    hands.append(([moved, draw()], 1.0, split_ace))
                    hands[i] = (cards, bet, split_ace)
                    if split_ace and not r.hit_split_aces:
                        break
            settled.append((cards, bet))
            i += 1

        running += _HILO[hole]          # dealer reveals
        while True:
            total, soft = hand_total(dealer)
            if total > 21 or total > 17:
                break
            if total == 17 and (r.dealer_stands_soft_17 or not soft):
                break
            dealer.append(draw())
        dealer_total, _ = hand_total(dealer)

        net = 0.0
        for cards, bet in settled:
            total, _ = hand_total(cards)
            if total > 21:
                net -= bet
            elif dealer_total > 21 or total > dealer_total:
                net += bet
            elif total < dealer_total:
                net -= bet
        return net, running

    # -- reporting -----------------------------------------------------
    def cell_actions(self, total: int, up: int) -> List[Tuple[int, str, int]]:
        """Greedy action and sample count per bucket for a hard cell."""
        out = []
        for b in range(self.tc_min, self.tc_max + 1):
            s: CState = ("hard", total, up, b)
            acts = [a for a in ("H", "S", "D", "P") if self.q_n[(s, a)] > 0]
            n = sum(self.q_n[(s, a)] for a in ("H", "S", "D", "P"))
            out.append((b, max(acts, key=lambda a: self.q(s, a)) if acts else "?", n))
        return out

    def index_report(self, min_n: int = 500) -> List[str]:
        """Learned action per bucket for each Illustrious 18 cell, with
        the inferred flip point vs. the book index. Buckets with fewer
        than ``min_n`` visits are shown lowercase and ignored when
        inferring the flip (one starved bucket shouldn't poison a row)."""
        lines = []
        header = "                 TC " + " ".join(
            f"{b:>2d}" for b in range(self.tc_min, self.tc_max + 1))
        lines.append(header)
        for (total, up), (idx, hi, lo) in sorted(DEVIATIONS.items()):
            row = self.cell_actions(total, up)
            acts = " ".join(
                f"{(a if n >= min_n else a.lower()):>2s}" for _, a, n in row)
            solid = [(b, a) for b, a, n in row if n >= min_n]
            flip = next((b for b, a in solid
                         if a == hi and all(a2 == hi for b2, a2 in solid if b2 >= b)),
                        None)
            book = f"book {hi} at TC>={idx:+d}"
            learned = f"learned flip at TC>={flip:+d}" if flip is not None \
                else "learned: no clean flip yet"
            lines.append(f"hard {total:2d} vs {up:2d}:  {acts}   {book}; {learned}")
        return lines


# ---------------------------------------------------------------------------
# Greedy-evaluation refinement: paired index resolver.
# ---------------------------------------------------------------------------


class IndexResolver:
    """Refine learned index plays by direct paired EV measurement.

    On-policy epsilon-greedy MC control converges to the value of the
    *exploring* policy: hitting leads to more decisions, each carrying an
    epsilon chance of a random blunder, so continuation actions are
    undervalued relative to terminal ones (stand/double) and learned flip
    points skew low. This resolver removes that bias:

    * the trained learner's policy is frozen and followed greedily
      (epsilon = 0) for all continuation decisions;
    * for each Illustrious 18 cell and each true-count bucket, the exact
      hand is forced and BOTH candidate actions are played out from the
      same shuffled shoe (common random numbers), conditioned on the
      dealer peek showing no blackjack — the actual decision context;
    * the reported quantity is the paired mean of net(hi) - net(lo) per
      bucket. The learned index is where that difference crosses zero.

    The pairing cancels most shoe-composition variance, so a few tens of
    thousands of pairs per bucket resolves EV gaps of a few hundredths
    of a percent — gaps blanket MC needs hundreds of millions of
    episodes to see.
    """

    def __init__(self, learner: MCDeviationLearner):
        self.learner = learner
        self.rules = learner.rules
        self.rng = learner.rng

    # -- frozen greedy continuation -------------------------------------
    def _greedy(self, cards: List[int], up: int, bucket: int,
                min_n: int = 100) -> str:
        """Continuation action from the learner's Q, gated for robustness.

        The trained Q is trusted only where BOTH actions have real sample
        mass; otherwise fall back to the bust-safe default (stand 17+).
        Without the gate, a state whose only visited action is 'hit'
        (e.g. hard 20 in a sparse bucket) makes the hit arm self-destruct,
        systematically flattering the terminal arm it's paired against.
        """
        total, soft = hand_total(cards)
        kind = "soft" if soft else "hard"
        s: CState = (kind, total, up, bucket)
        default = "S" if total >= 17 else "H"
        other = "H" if default == "S" else "S"
        L = self.learner
        if (L.q_n[(s, "H")] >= min_n and L.q_n[(s, "S")] >= min_n
                and L.q(s, other) > L.q(s, default)):
            return other
        return default

    # -- forced-hand setup ------------------------------------------------
    def _partial_shuffle(self, shoe: List[int], k: int = 40) -> None:
        """Freshen the draw order cheaply: Fisher-Yates over the last k
        positions (the only cards a pair can consume), each swapped with
        a uniformly random position in the whole shoe."""
        rng = self.rng
        n = len(shoe)
        for i in range(n - 1, max(n - 1 - k, 0), -1):
            j = rng.randrange(0, i + 1)
            shoe[i], shoe[j] = shoe[j], shoe[i]

    def _setup_from(self, base: List[int], base_running: int,
                    total: int, up: int) -> Tuple[List[int], List[int], int] | None:
        """Copy the base shoe, freshen its tail, force the cell's cards,
        ensure a non-blackjack hole. Returns (shoe, player, running)."""
        shoe = list(base)
        self._partial_shuffle(shoe)
        running = base_running
        c1 = self.rng.randint(max(2, total - 10), min(10, total - 2))
        c2 = total - c1
        for c in (c1, c2, up):
            try:
                shoe.remove(c)
            except ValueError:
                return None
            running += _HILO[c]
        if not shoe:
            return None
        # Dealer peek: condition on no blackjack. The BJ-making hole is
        # swapped with a uniformly random eligible position — swapping
        # with a nearby card would park a ten right where the next draw
        # comes from, enriching first draws with tens and biasing both
        # arms (this exact bug produced +27% phantom doubling deltas).
        if up >= 10 and shoe[-1] + up == 21:
            eligible = [j for j in range(len(shoe) - 1) if shoe[j] + up != 21]
            if not eligible:
                return None
            j = self.rng.choice(eligible)
            shoe[-1], shoe[j] = shoe[j], shoe[-1]
        return shoe, [c1, c2], running

    def _finish(self, cards: List[int], bet: float, shoe: List[int],
                up: int, hole: int, running: int) -> float:
        """Dealer plays; settle one hand."""
        r = self.rules
        total, _ = hand_total(cards)
        if total > 21:
            return -bet
        dealer = [up, hole]
        while True:
            dt, soft = hand_total(dealer)
            if dt > 21 or dt > 17:
                break
            if dt == 17 and (r.dealer_stands_soft_17 or not soft):
                break
            dealer.append(shoe.pop())
        dt, _ = hand_total(dealer)
        if dt > 21 or total > dt:
            return bet
        if total < dt:
            return -bet
        return 0.0

    def _play_arm(self, action: str, player: List[int], shoe: List[int],
                  up: int, running: int, bucket: int) -> float:
        """Play one candidate action, then frozen-greedy to completion."""
        cards = list(player)
        shoe = list(shoe)                 # each arm consumes its own copy
        hole = shoe.pop()
        bet = 1.0
        run = running

        def draw() -> int:
            nonlocal run
            c = shoe.pop()
            run += _HILO[c]
            return c

        if action == "D":
            bet = 2.0
            cards.append(draw())
        elif action == "H":
            cards.append(draw())
            while True:
                total, _ = hand_total(cards)
                if total >= 21:
                    break
                b = self.learner._bucket(run / max(len(shoe) / 52.0, 0.5))
                if self._greedy(cards, up, b) != "H":
                    break
                cards.append(draw())
        # "S": no cards drawn.
        return self._finish(cards, bet, shoe, up, hole, run)

    # -- resolution --------------------------------------------------------
    def resolve_cell(self, total: int, up: int, hi: str,
                     pairs: int) -> Dict[int, Tuple[float, int, float]]:
        """Per realized bucket: (mean paired delta net(hi)-net(H), n, SE)."""
        sums: Dict[int, float] = defaultdict(float)
        sq: Dict[int, float] = defaultdict(float)
        ns: Dict[int, int] = defaultdict(int)
        L = self.learner
        target = pairs * (L.tc_max - L.tc_min + 1)
        done = 0
        base: List[int] = []
        base_running = 0
        since_rebuild = 0
        while done < target:
            if not base or since_rebuild >= 50:
                base, base_running = L._stratified_shoe()
                since_rebuild = 0
            since_rebuild += 1
            setup = self._setup_from(base, base_running, total, up)
            done += 1                     # count attempts: no infinite loops
            if setup is None:
                since_rebuild = 50        # infeasible base; force rebuild
                continue
            shoe, player, running = setup
            b = L._bucket(running / max(len(shoe) / 52.0, 0.5))
            net_hi = self._play_arm(hi, player, shoe, up, running, b)
            net_lo = self._play_arm("H", player, shoe, up, running, b)
            d = net_hi - net_lo
            sums[b] += d
            sq[b] += d * d
            ns[b] += 1
        out = {}
        for b in ns:
            n = ns[b]
            mean = sums[b] / n
            var = max(sq[b] / n - mean * mean, 0.0)
            out[b] = (mean, n, (var / n) ** 0.5)
        return out

    def report(self, pairs: int = 20_000) -> List[str]:
        lines = []
        L = self.learner
        header = "                 TC " + " ".join(
            f"{b:>6d}" for b in range(L.tc_min, L.tc_max + 1))
        lines.append(header + "   (paired dEV, % of a unit)")
        agree = 0
        for (total, up), (idx, hi, lo) in sorted(DEVIATIONS.items()):
            table = self.resolve_cell(total, up, hi, pairs)
            cells = []
            flip = None
            for b in range(L.tc_min, L.tc_max + 1):
                if b in table:
                    d, _n, _se = table[b]
                    cells.append(f"{100*d:+6.2f}")
                    if flip is None and d > 0 and all(
                            table[b2][0] > 0 for b2 in table if b2 >= b):
                        flip = b
                else:
                    cells.append("     .")
            if flip is not None:
                d, _n, se = table[flip]
                sig = "" if d >= 2 * se else "?"    # thin gap: not yet resolved
                learned = f"flip {flip:+d}{sig}"
            else:
                learned = "no flip"
            mark = ""
            if flip is not None and abs(flip - idx) <= 1:
                agree += 1
                mark = "  *"
            lines.append(f"hard {total:2d} vs {up:2d}: " + " ".join(cells)
                         + f"   book {idx:+d}, {learned}{mark}")
        lines.append(f"\n{agree}/{len(DEVIATIONS)} indices within one bucket"
                     " of book (*); '?' = zero-crossing thinner than 2 SE,"
                     " raise --refine-pairs to resolve")
        return lines
