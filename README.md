# blackjack-sim

Monte Carlo blackjack: basic strategy, Hi-Lo card counting, and MIT-style
bet sizing — with two learners that derive the strategy and the bet ramp
from simulation instead of taking them on faith.

Pure Python stdlib. No dependencies. ~75k rounds/sec in CPython.

## Quick start

```bash
# Full MIT playbook: bet spread + Illustrious 18 deviations + wong out
python mc.py simulate --shoes 5000 --bet spread --deviations --wong-out -1

# Baseline control: flat-bet basic strategy (should show ~-0.5% edge)
python mc.py simulate --shoes 5000 --bet flat

# Learn basic strategy from scratch via Monte Carlo control
python mc.py learn-strategy --episodes 5000000

# Measure edge per true count and derive a Kelly bet ramp from the data
python mc.py learn-betting --shoes 20000

# Learn the Illustrious 18 index plays from scratch (overnight-scale run)
python mc.py learn-deviations --episodes 100000000

# S17 vs H17 rule cost, same seed and bet sizer
python mc.py --seed 42 compare --shoes 40000 --bet flat

# Bankroll trajectory chart (SVG, open in a browser) + per-round CSV
python mc.py --seed 42 simulate --shoes 3000 --bet kelly --deviations \
    --wong-out -1 --bankroll 50000 --plot bankroll.svg --csv rounds.csv

# Sanity tests
python tests/test_core.py
```

```bash
# MIT team structure: 4 spotters flat-betting, one Big Player called in hot
python mc.py --seed 42 team --ticks 100000 --tables 4 --bps 1 --call-in 2 \
    --bankroll 100000 --plot team.svg
```

## Layout

```
blackjack/
  cards.py      int-valued cards & shoe (no suit/face objects — MC speed)
  rules.py      table rules (6D, S17, DAS, 3:2, 75% pen by default)
  counting.py   Hi-Lo running count -> true count (per remaining deck)
  strategy.py   book basic strategy + Illustrious 18 + insurance index
  betting.py    flat / spread ramp / fractional Kelly, wong-out support
  engine.py     print-free simulation loop, per-round records
  stats.py      EV, variance, edge-by-TC, risk of ruin, Kelly ramp fit
  learn.py      MC control (learn strategy) + bet-ramp estimation
  team.py       spotters + Big Player(s), shared bankroll, call-in/leave
  plot.py       bankroll trajectory -> CSV / stdlib SVG chart
mc.py           CLI: simulate | compare | learn-strategy | learn-betting
legacy/BJ.py    original simulator, kept for reference
tests/          sanity tests (hand math, indices, known-edge check)
```

## What changed vs. legacy/BJ.py

1. **True count fixed.** The legacy code accumulated a running count but
   used it as a true count. Bet ramps and index plays are calibrated to
   running count *per remaining deck*; without the division the bet
   trigger fires on the wrong signal. `HiLoCounter.true_count()` divides
   by decks remaining.
2. **Play deviations added.** Counting only moved the legacy bet, never
   the play. The Illustrious 18 index plays plus insurance at TC >= +3
   are where a large share of a counter's edge lives; both are in
   `strategy.py` behind `--deviations`.
3. **MIT-style bet sizing.** `SpreadBet` (discrete 1-12 unit ramp),
   `KellyBet` (bankroll-fraction, half-Kelly default), and wong-out —
   while sat out, the engine still burns cards so the count keeps evolving.
4. **Hot path stripped.** No prints, no card objects, no numpy (the
   `Aces()` permutation table is replaced by a 3-line ace demotion loop).
   Result: ~75k rounds/sec, enough for tight confidence intervals.
5. **Rule/payout bugs fixed.** Doubled busts now settle as double losses
   (legacy tagged them `Result(2)`/DDWIN); 21 after a split is no longer
   paid as a natural; split aces receive one card; dealer peek and
   insurance are modeled.
6. **Learning, not just replay.** `learn-strategy` runs first-visit MC
   control (epsilon-greedy, tabular Q) and diffs the learned policy
   against the book — agreement climbs with episodes and the stragglers
   are the near-EV-tie soft doubles, as expected. `learn-betting`
   measures realized edge per true-count bucket and fits a
   Kelly-proportional ramp, reproducing the count->bet card from data.
   `learn-deviations` closes the loop: MC control with the true count in
   the state, dealt from count-stratified shoes (extreme counts are rare
   in nature, so shoes are constructed to target running counts and the
   count then evolves honestly within them). With enough episodes the
   greedy action flips at the book index — the Illustrious 18 derived
   from payouts alone. At ~12M episodes several indices land exactly
   (12v3 at +2, 10vT at +4, 12v4 at 0); doubling indices resolve slowest
   (double variance, thin EV gaps). The `--refine-pairs` pass then fixes
   the residual on-policy bias (exploration contaminates continuation
   actions, flattering terminal ones): it freezes the learned policy and
   directly measures the paired EV difference between the candidate
   actions from shared shoes, per cell per bucket, conditioned on the
   dealer peek. Zero-crossing = learned index. 12/16 indices land within
   one bucket of book at 4M episodes + 12k pairs; '?'-flagged cells have
   gaps thinner than 2 SE and want more pairs.

## Reference numbers (seeded runs, 6D S17 DAS 75% pen)

| Configuration                          | Edge (% of action) |
|----------------------------------------|--------------------|
| Flat bet, basic strategy               | ~ -0.5%            |
| Spread 1-12, no deviations             | ~ +0.5 to +1.0%    |
| Spread 1-12 + I18 + wong out at TC<-1  | ~ +1.0 to +1.5%    |
| H17 rule change (vs S17 baseline)      | ~ -0.25% penalty   |
| Team: 4 tables + 1 BP (BP action only) | ~ +3% of BP action |

The team simulation charges the spotters' flat-bet grind (~-0.5% of their
action) against the shared bankroll and lets the Big Player play only
called-in shoes with Kelly sizing on the team roll — the bet-to-count
correlation that identifies a solo counter never appears on any one seat.
Same seed comparison: solo Kelly counter ~2.8% risk of ruin; the team,
with its diversified income across tables, ~0.1%.

`BasicStrategy(h17=True)` applies the standard H17 chart changes (11 vs A
double, soft 18 vs 2 double, soft 19 vs 6 double); `compare` runs both
games on the same seed with the matching chart.

Risk-of-ruin output uses the diffusion approximation
`exp(-2 * EV * bankroll / variance)`; treat it as a planning number, not
a guarantee.
