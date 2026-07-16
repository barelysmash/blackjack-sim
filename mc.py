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
from blackjack.plot import write_csv, write_svg
from blackjack.rules import Rules
from blackjack.stats import by_true_count, risk_of_ruin, summarize
from blackjack.strategy import BasicStrategy


def _rules(args) -> Rules:
    return Rules(
        num_decks=args.decks,
        dealer_stands_soft_17=not args.h17,
        penetration=args.penetration,
    )


def _betting(args):
    if args.bet == "flat":
        return FlatBet()
    if args.bet == "spread":
        return SpreadBet(wong_out_below=args.wong_out)
    return KellyBet(wong_out_below=args.wong_out)


def cmd_simulate(args) -> None:
    rules = _rules(args)
    strategy = BasicStrategy(use_deviations=args.deviations, h17=args.h17)
    betting = _betting(args)

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

    if args.csv:
        write_csv(res.records, args.csv)
        print(f"\nwrote {args.csv}")
    if args.plot:
        rule = "H17" if args.h17 else "S17"
        write_svg(
            res.records, args.plot,
            title=f"Bankroll — {args.bet}"
                  + (" + I18" if args.deviations else "")
                  + f", {args.decks}D {rule}",
            start_bankroll=args.bankroll,
        )
        print(f"wrote {args.plot} (open in a browser)")


def cmd_compare(args) -> None:
    """Same seed, same bet sizer: S17 vs H17 side by side.

    The edge is measured with an unconstrained bankroll (otherwise a ruin
    truncates the sample and caps how much data a longer run can add);
    risk of ruin is then computed analytically for --bankroll.
    """
    rows = []
    for label, s17 in (("S17", True), ("H17", False)):
        rules = Rules(num_decks=args.decks, dealer_stands_soft_17=s17,
                      penetration=args.penetration)
        strategy = BasicStrategy(use_deviations=args.deviations, h17=not s17)
        sim = Simulator(rules, strategy, _betting(args), seed=args.seed)
        res = sim.run(args.shoes, bankroll=1e12,
                      min_bet=args.min_bet, stop_on_ruin=False)
        s = summarize(res.records, args.min_bet)
        ror = risk_of_ruin(s["ev_per_round"], s["sd_per_round"], args.bankroll)
        rows.append((label, s, ror))

    hdr = f"{'':6s}{'rounds':>10s}{'edge %':>9s}{'EV/round':>11s}{'±95%':>8s}{'total net $':>13s}{'RoR %':>8s}"
    print(hdr)
    for label, s, ror in rows:
        print(f"{label:6s}{s['rounds']:>10,}{s['edge_pct_of_action']:>+9.3f}"
              f"{s['ev_per_round']:>+11.3f}{1.96*s['se_ev']:>8.3f}"
              f"{s['total_net']:>13,.0f}{100*ror:>8.1f}")
    d = rows[1][1]["edge_pct_of_action"] - rows[0][1]["edge_pct_of_action"]
    print(f"\nH17 rule cost: {d:+.3f}% of action "
          f"(book value is roughly -0.2%; RoR is for a ${args.bankroll:,.0f} bankroll)")


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


def cmd_learn_deviations(args) -> None:
    from blackjack.learn import MCDeviationLearner
    learner = MCDeviationLearner(_rules(args), seed=args.seed)
    t0 = time.time()
    step = max(args.episodes // 10, 1)
    done = 0
    while done < args.episodes:
        n = min(step, args.episodes - done)
        learner.train(n)
        done += n
        row = learner.cell_actions(16, 10)
        acts = " ".join(a for _, a, _ in row)
        print(f"episodes {done:>12,}  eps={learner.epsilon:.3f}  "
              f"hard 16 vs T by TC [{learner.tc_min}..{learner.tc_max}]: {acts}")
    print(f"\ndone in {time.time()-t0:.0f}s — learned index plays vs book:\n")
    for line in learner.index_report():
        print(line)
    print("\nNote: doubling indices (9v2, 10vT, 11vA, ...) and negative-index")
    print("stands are the slowest to resolve; disagreements at low n are")
    print("sampling noise, not engine error. More episodes tightens them.")
    if args.refine_pairs > 0:
        from blackjack.learn import IndexResolver
        print(f"\nrefinement pass: paired greedy evaluation "
              f"({args.refine_pairs:,} pairs/cell-bucket)...")
        t1 = time.time()
        resolver = IndexResolver(learner)
        for line in resolver.report(pairs=args.refine_pairs):
            print(line)
        print(f"refinement took {time.time()-t1:.0f}s")


def cmd_team(args) -> None:
    from blackjack.betting import KellyBet, SpreadBet
    from blackjack.stats import risk_of_ruin
    from blackjack.team import TeamSimulator

    if args.bet == "kelly":
        bp_bet = KellyBet(fraction=0.5, max_units=args.max_units)
    else:
        bp_bet = SpreadBet()
    team = TeamSimulator(
        _rules(args), num_tables=args.tables, num_bps=args.bps,
        call_in_tc=args.call_in, leave_tc=args.leave,
        bp_betting=bp_bet, seed=args.seed)
    t0 = time.time()
    res = team.run(args.ticks, bankroll=args.bankroll,
                   spotter_bet=args.min_bet)
    dt = time.time() - t0

    n = res.ticks
    print(f"ticks (rounds/table) {n:,}   tables {args.tables}   BPs {args.bps}"
          f"   in {dt:.1f}s")
    print(f"call-ins             {res.call_ins:,}  "
          f"(BP utilization {100*res.bp_utilization:.1f}% of ticks)")
    se = 0.0
    if res.records:
        nets = [r.net for r in res.records]
        mean = sum(nets) / n
        var = sum((x - mean) ** 2 for x in nets) / max(n - 1, 1)
        se = (var / n) ** 0.5
        sp_edge = 100 * res.spotter_net / max(res.spotter_action, 1)
        bp_edge = 100 * res.bp_net / max(res.bp_action, 1)
        print(f"spotter grind        {res.spotter_rounds:,} rounds, "
              f"net ${res.spotter_net:,.0f} ({sp_edge:+.2f}% of action)")
        print(f"BP hands             {res.bp_rounds:,} rounds, "
              f"net ${res.bp_net:,.0f} ({bp_edge:+.2f}% of action), "
              f"avg bet ${res.bp_action/max(res.bp_rounds,1):,.0f}")
        print(f"team net             ${res.team_net:,.0f}"
              + ("  ** RUINED **" if res.ruined else ""))
        print(f"EV/tick              ${mean:+.2f} (± {1.96*se:.2f} at 95%)")
        ror = risk_of_ruin(mean, var ** 0.5, args.bankroll)
        print(f"team risk of ruin    {100*ror:.1f}% (diffusion approx, "
              f"${args.bankroll:,.0f} bankroll)")
        print(f"final bankroll       ${res.records[-1].bankroll:,.0f}")
        if args.plot:
            write_svg(res.records, args.plot,
                      title=f"Team bankroll — {args.tables} tables, "
                            f"{args.bps} BP, call-in TC {args.call_in:+g}",
                      start_bankroll=args.bankroll)
            print(f"wrote {args.plot}")


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
    ps.add_argument("--csv", type=str, default=None,
                    help="write per-round trajectory to this CSV path")
    ps.add_argument("--plot", type=str, default=None,
                    help="write bankroll trajectory SVG to this path")
    ps.set_defaults(func=cmd_simulate)

    pc = sub.add_parser("compare", help="S17 vs H17, same seed and bet sizer")
    pc.add_argument("--shoes", type=int, default=5_000)
    pc.add_argument("--bankroll", type=float, default=10_000)
    pc.add_argument("--min-bet", type=float, default=25)
    pc.add_argument("--bet", choices=["flat", "spread", "kelly"], default="flat")
    pc.add_argument("--deviations", action="store_true")
    pc.add_argument("--wong-out", type=float, default=None)
    pc.set_defaults(func=cmd_compare)

    pl = sub.add_parser("learn-strategy")
    pl.add_argument("--episodes", type=int, default=1_000_000)
    pl.set_defaults(func=cmd_learn_strategy)

    pb = sub.add_parser("learn-betting")
    pb.add_argument("--shoes", type=int, default=5_000)
    pb.set_defaults(func=cmd_learn_betting)

    pd = sub.add_parser("learn-deviations",
                        help="learn Illustrious 18 index plays from scratch")
    pd.add_argument("--episodes", type=int, default=20_000_000)
    pd.add_argument("--refine-pairs", type=int, default=20_000,
                    help="paired greedy-evaluation samples per cell-bucket "
                         "after training (0 to disable)")
    pd.set_defaults(func=cmd_learn_deviations)

    pt = sub.add_parser("team", help="MIT-style team: spotters + Big Player(s)")
    pt.add_argument("--ticks", type=int, default=100_000,
                    help="rounds dealt per table")
    pt.add_argument("--tables", type=int, default=4)
    pt.add_argument("--bps", type=int, default=1)
    pt.add_argument("--call-in", type=float, default=2.0)
    pt.add_argument("--leave", type=float, default=1.0)
    pt.add_argument("--bankroll", type=float, default=100_000)
    pt.add_argument("--min-bet", type=float, default=25)
    pt.add_argument("--bet", choices=["kelly", "spread"], default="kelly")
    pt.add_argument("--max-units", type=float, default=20)
    pt.add_argument("--plot", type=str, default=None)
    pt.set_defaults(func=cmd_team)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
