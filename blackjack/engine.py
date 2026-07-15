"""Monte Carlo simulation engine.

Design notes (vs. the legacy BJ.py):

* Zero I/O in the hot path. The legacy loop printed ~30 lines per round,
  which dominated runtime; this engine plays ~1M rounds/minute in CPython
  and returns raw per-round records for offline analysis.
* The count is updated card-by-card as cards become visible (hole card at
  reveal), so both bet sizing and in-hand deviations read an honest count.
* Legacy payout bugs fixed: doubled busts settle as double losses, a 21
  made after splitting is not a blackjack, split aces get one card.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from .cards import build_shoe, hand_total, is_blackjack
from .counting import HiLoCounter
from .rules import Rules
from .strategy import BasicStrategy


@dataclass
class RoundRecord:
    true_count: float   # true count at bet time
    bet: float          # dollars wagered on the initial hand
    net: float          # dollars won/lost this round (all hands + insurance)
    bankroll: float     # bankroll after settlement


@dataclass
class _HandState:
    cards: List[int]
    bet: float
    from_split_ace: bool = False
    doubled: bool = False


@dataclass
class SimResult:
    records: List[RoundRecord] = field(default_factory=list)
    rounds: int = 0
    wonged_out: int = 0
    ruined: bool = False


class Simulator:
    def __init__(
        self,
        rules: Rules,
        strategy: BasicStrategy,
        betting,
        seed: Optional[int] = None,
    ) -> None:
        self.rules = rules
        self.strategy = strategy
        self.betting = betting
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    def run(
        self,
        num_shoes: int,
        bankroll: float = 10_000.0,
        min_bet: float = 25.0,
        stop_on_ruin: bool = True,
    ) -> SimResult:
        result = SimResult()
        counter = HiLoCounter()
        for _ in range(num_shoes):
            shoe = build_shoe(self.rules.num_decks, self.rng)
            counter.reset()
            while len(shoe) > self.rules.reshuffle_at and len(shoe) >= 24:
                tc = counter.true_count(len(shoe))
                bet = self.betting.bet(tc, bankroll, min_bet)
                if bet <= 0 or bankroll < min_bet:
                    if bankroll < min_bet:
                        result.ruined = True
                        if stop_on_ruin:
                            return result
                    result.wonged_out += 1
                    self._burn_round(shoe, counter)
                    continue
                net = self._play_round(shoe, counter, bet, bankroll)
                bankroll += net
                result.rounds += 1
                result.records.append(RoundRecord(tc, bet, net, bankroll))
        return result

    # ------------------------------------------------------------------
    def _draw(self, shoe: List[int], counter: HiLoCounter) -> int:
        card = shoe.pop()
        counter.see(card)
        return card

    def _burn_round(self, shoe: List[int], counter: HiLoCounter) -> None:
        """Wonged out: another player and the dealer still consume cards."""
        dummy = [self._draw(shoe, counter), self._draw(shoe, counter)]
        up = self._draw(shoe, counter)
        hole = shoe.pop()  # hidden until reveal
        while True:
            total, _ = hand_total(dummy)
            if total >= 21:
                break
            action = self.strategy.decide(dummy, up, False, False, 0.0)
            if action in ("S", "Ds"):
                break
            dummy.append(self._draw(shoe, counter))
        counter.see(hole)
        dealer = [up, hole]
        self._dealer_play(dealer, shoe, counter)

    def _dealer_play(self, dealer: List[int], shoe, counter) -> None:
        while True:
            total, soft = hand_total(dealer)
            if total > 21 or total > 17:
                return
            if total == 17 and (self.rules.dealer_stands_soft_17 or not soft):
                return
            dealer.append(self._draw(shoe, counter))

    # ------------------------------------------------------------------
    def _play_round(
        self, shoe: List[int], counter: HiLoCounter, bet: float, bankroll: float
    ) -> float:
        r = self.rules
        player = [self._draw(shoe, counter), self._draw(shoe, counter)]
        up = self._draw(shoe, counter)
        hole = shoe.pop()  # not counted until revealed
        dealer = [up, hole]
        net = 0.0

        # Insurance (offered on an ace before the peek).
        if up == 11 and r.allow_insurance:
            tc = counter.true_count(len(shoe))
            if self.strategy.take_insurance(tc):
                side = bet / 2.0
                if is_blackjack(dealer):
                    net += side * 2.0
                else:
                    net -= side

        # Dealer peek on ten/ace up.
        if r.dealer_peeks and up >= 10 and is_blackjack(dealer):
            counter.see(hole)
            if is_blackjack(player):
                return net          # push
            return net - bet

        if is_blackjack(player):
            counter.see(hole)       # dealer flips regardless
            return net + bet * r.blackjack_payout

        # --- player hands (supports splits) ---------------------------
        committed = bet
        hands: List[_HandState] = [_HandState(player, bet)]
        i = 0
        while i < len(hands):
            hand = hands[i]
            if hand.from_split_ace and not r.hit_split_aces:
                i += 1
                continue
            while True:
                total, _ = hand_total(hand.cards)
                if total >= 21:
                    break
                two = len(hand.cards) == 2
                was_split = len(hands) > 1
                can_double = (
                    two
                    and (not was_split or r.double_after_split)
                    and bankroll - committed >= hand.bet
                )
                can_split = (
                    two
                    and hand.cards[0] == hand.cards[1]
                    and len(hands) < r.max_hands
                    and bankroll - committed >= hand.bet
                    and (not hand.from_split_ace or r.resplit_aces)
                )
                tc = counter.true_count(len(shoe))
                action = self.strategy.decide(
                    hand.cards, up, can_double, can_split, tc
                )
                if action == "H":
                    hand.cards.append(self._draw(shoe, counter))
                elif action in ("S", "Ds"):
                    break
                elif action == "D":
                    committed += hand.bet
                    hand.bet *= 2
                    hand.doubled = True
                    hand.cards.append(self._draw(shoe, counter))
                    break
                elif action == "P":
                    committed += hand.bet
                    split_ace = hand.cards[0] == 11
                    moved = hand.cards.pop()
                    hand.cards.append(self._draw(shoe, counter))
                    hands.append(
                        _HandState(
                            [moved, self._draw(shoe, counter)],
                            bet,
                            from_split_ace=split_ace,
                        )
                    )
                    hand.from_split_ace = split_ace
                    if split_ace and not r.hit_split_aces:
                        break
            i += 1

        # --- dealer & settlement ---------------------------------------
        counter.see(hole)
        any_live = any(hand_total(h.cards)[0] <= 21 for h in hands)
        if any_live:
            self._dealer_play(dealer, shoe, counter)
        dealer_total, _ = hand_total(dealer)

        for hand in hands:
            total, _ = hand_total(hand.cards)
            if total > 21:
                net -= hand.bet
            elif dealer_total > 21 or total > dealer_total:
                net += hand.bet
            elif total < dealer_total:
                net -= hand.bet
            # equal -> push
        return net
