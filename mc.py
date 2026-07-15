#!/usr/bin/env python3
"""blackjack-sim command line.

Subcommands
-----------
simulate        Monte Carlo EV/bankroll simulation with a chosen bet sizer.
learn-strategy  Learn basic strategy from scratch via MC control; diff vs book.
learn-betting   Measure edge per true count; derive a Kelly bet ramp.

Examples
--------
python mc.py simulate --shoes 5000 --bet spread --deviations
python mc.py simulate --shoes 5000 --bet kelly --wong-out -1
python mc.py learn-strategy --episodes 2000000
python mc.py learn-betting --shoes 20000
"""

from __future__ import annotations

import argparse
import time

from blackjack.betting import FlatBet, KellyBet, SpreadBet
from blackjack.engine import Simulator
from blackjack.learn import MCStrategyLearner, learn_bet_ramp
from blackjack.rules import Rules
from blackjack.stats import by_true_count, risk_of_ruin, summarize
from blackjack.strategy import BasicStrategy


def _rules(args) -> Rules:
    return Rules(
        num_decks=args.decks,
        dealer_stands_soft_17=not args.h17,
        penetration=args.penetration,
    )


def cmd_simulate(args) -> None:
    rules = _rules(args)
    strategy = BasicStrategy(use_deviations=args.deviations)
    if args.bet == "flat":
        betting = FlatBet()
    elif args.bet == "spread":
        betting = SpreadBet(wong_out_below=args.wong_out)
    else:
        betting = KellyBet(wong_out_below=args.wong_out)

    sim = Simulator(rules, strategy, betting, seed=args.seed)
    t0 = time.time()
    res = sim.run(args.shoes, bankroll=args.bankroll, min_bet=args.min_bet,
                  stop_on_ruin=not args.no_ruin_stop)
    dt = time.time() - t0

    s = summarize(res.records, args.min_bet)
    print(f"rounds played      {s['rounds']:,}  "
          f"(wonged out {res.wonged_out:,})  in {dt:.1f}s "
          f"({s['rounds']/dt:,.0f} rounds/s)")
    if s["rounds"] == 0:
        return
    print(f"total net          ${s['total_net']:,.0f}")
    print(f"final bankroll     ${s['final_bankroll']:,.0f}"
          + ("  ** RUINED **" if res.ruined else ""))
    print(f"avg bet            ${s['avg_bet']:,.2f}")
    print(f"EV/round           ${s['ev_per_round']:+.3f} "
          f"(± {1.96*s['se_ev']:.3f} at 95%)")
    print(f"edge (% of action) {s['edge_pct_of_action']:+.3f}%")
    print(f"std dev/round      ${s['sd_per_round']:.2f}")
    ror = risk_of_ruin(s["ev_per_round"], s["sd_per_round"], args.bankroll)
    print(f"risk of ruin       {100*ror:.1f}% (diffusion approx)")
    print("\nedge by true count (at bet time):")
    for tc, v in by_true_count(res.records).items():
        print(f"  TC {tc:+d}: {v['edge_pct']:+7.3f}%   (n={v['rounds']:,})")


def cmd_learn_strategy(args) -> None:
    learner = MCStrategyLearner(_rules(args), seed=args.seed)
    t0 = time.time()
    step = max(args.episodes // 10, 1)
    done = 0
    while done < args.episodes:
        n = min(step, args.episodes - done)
        learner.train(n)
        done += n
        matches, total, _ = learner.compare_to_book()
        pct = 100 * matches / total if total else 0.0
        print(f"episodes {done:>10,}  agreement with book: "
              f"{matches}/{total} ({pct:.1f}%)  eps={learner.epsilon:.3f}")
    matches, total, diffs = learner.compare_to_book()
    print(f"\nfinal agreement: {matches}/{total} in {time.time()-t0:.0f}s")
    if diffs:
        print("disagreements (mostly low-n or near-EV-tie states):")
        for line in diffs:
            print("  " + line)


def cmd_learn_betting(args) -> None:
    table, ramp = learn_bet_ramp(args.shoes, _rules(args), seed=args.seed)
    print("measured edge per unit wagered, by true count:")
    for tc, v in table.items():
        print(f"  TC {tc:+d}: {v['edge_pct']:+7.3f}%   (n={v['rounds']:,})")
    print("\nderived Kelly-proportional bet ramp (units):")
    for tc, units in sorted(ramp.items()):
        print(f"  TC {tc:+d}: {units:g}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decks", type=int, default=6)
    p.add_argument("--h17", action="store_true", help="dealer hits soft 17")
    p.add_argument("--penetration", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("simulate")
    ps.add_argument("--shoes", type=int, default=1_000)
    ps.add_argument("--bankroll", type=float, default=10_000)
    ps.add_argument("--min-bet", type=float, default=25)
    ps.add_argument("--bet", choices=["flat", "spread", "kelly"], default="spread")
    ps.add_argument("--deviations", action="store_true",
                    help="apply Illustrious 18 index plays + insurance index")
    ps.add_argument("--wong-out", type=float, default=None,
                    help="sit out when true count is below this value")
    ps.add_argument("--no-ruin-stop", action="store_true")
    ps.set_defaults(func=cmd_simulate)

    pl = sub.add_parser("learn-strategy")
    pl.add_argument("--episodes", type=int, default=1_000_000)
    pl.set_defaults(func=cmd_learn_strategy)

    pb = sub.add_parser("learn-betting")
    pb.add_argument("--shoes", type=int, default=5_000)
    pb.set_defaults(func=cmd_learn_betting)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
