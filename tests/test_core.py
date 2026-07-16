"""Sanity tests. Run: python -m pytest tests/ (or python tests/test_core.py)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blackjack.betting import FlatBet, SpreadBet
from blackjack.cards import hand_total, is_blackjack
from blackjack.counting import HiLoCounter
from blackjack.engine import Simulator
from blackjack.rules import Rules
from blackjack.strategy import BasicStrategy
from blackjack.stats import summarize


def test_hand_values():
    assert hand_total([11, 10]) == (21, True)
    assert hand_total([11, 11]) == (12, True)
    assert hand_total([11, 5, 10]) == (16, False)   # ace demoted
    assert hand_total([10, 10, 5]) == (25, False)
    assert is_blackjack([11, 10]) and not is_blackjack([7, 7, 7])


def test_true_count_normalizes_by_decks():
    c = HiLoCounter()
    for _ in range(6):
        c.see(5)                       # running +6
    assert c.true_count(6 * 52) == 1.0  # 6 decks left -> TC +1
    assert c.true_count(52) == 6.0      # 1 deck left  -> TC +6


def test_book_plays():
    s = BasicStrategy()
    assert s.decide([10, 6], 10, True, False) == "H"    # 16 v T: hit
    assert s.decide([8, 8], 10, True, True) == "P"      # always split 8s
    assert s.decide([5, 6], 5, True, False) == "D"      # 11 v 5: double
    assert s.decide([11, 7], 9, True, False) == "H"     # soft 18 v 9: hit
    assert s.decide([11, 7], 3, True, False) == "D"     # soft 18 v 3 (Ds)
    assert s.decide([11, 7], 3, False, False) == "S"    # Ds falls back to stand


def test_deviations():
    s = BasicStrategy(use_deviations=True)
    assert s.decide([10, 6], 10, True, False, true_count=1.0) == "S"   # I18
    assert s.decide([10, 6], 10, True, False, true_count=-1.0) == "H"
    assert s.take_insurance(3.2) and not s.take_insurance(2.0)


def test_simulation_edge_is_sane():
    """Flat-bet basic strategy should land near the known ~-0.5% edge."""
    sim = Simulator(Rules(), BasicStrategy(), FlatBet(), seed=7)
    res = sim.run(600, bankroll=1e9, min_bet=1.0, stop_on_ruin=False)
    s = summarize(res.records, 1.0)
    assert s["rounds"] > 15_000
    assert -1.5 < s["edge_pct_of_action"] < 0.5, s["edge_pct_of_action"]


def test_wong_out_burns_rounds():
    sim = Simulator(Rules(), BasicStrategy(),
                    SpreadBet(wong_out_below=99.0), seed=1)  # always out
    res = sim.run(3, bankroll=1e9, min_bet=1.0, stop_on_ruin=False)
    assert res.rounds == 0 and res.wonged_out > 0


def test_h17_adjustments():
    s17 = BasicStrategy()
    h17 = BasicStrategy(h17=True)
    assert s17.decide([6, 5], 11, True, False) == "H"    # 11 v A: hit (S17)
    assert h17.decide([6, 5], 11, True, False) == "D"    # 11 v A: double (H17)
    assert s17.decide([11, 7], 2, True, False) == "S"    # soft 18 v 2 (S17)
    assert h17.decide([11, 7], 2, True, False) == "D"    # soft 18 v 2 (H17)
    assert h17.decide([11, 8], 6, True, False) == "D"    # soft 19 v 6 (H17)
    assert h17.decide([11, 8], 6, False, False) == "S"   # Ds fallback


def test_plot_outputs():
    import os
    import tempfile
    from blackjack.plot import write_csv, write_svg
    tmp_dir = tempfile.gettempdir()
    sim = Simulator(Rules(), BasicStrategy(), FlatBet(), seed=2)
    res = sim.run(20, bankroll=1e6, min_bet=25.0, stop_on_ruin=False)
    csv_p, svg_p = os.path.join(tmp_dir, "bj.csv"), os.path.join(tmp_dir, "bj.svg")
    write_csv(res.records, csv_p)
    write_svg(res.records, svg_p, start_bankroll=1e6)
    assert open(csv_p).readline().startswith("round,")
    assert open(svg_p).read().lstrip().startswith("<svg")


def test_stratified_shoe_count_consistency():
    from blackjack.counting import _HILO
    from blackjack.learn import MCDeviationLearner
    learner = MCDeviationLearner(seed=9)
    for _ in range(50):
        shoe, running = learner._stratified_shoe()
        # Removed cards' Hi-Lo tags must sum to the reported running count.
        full = {v: 4 * 6 for v in range(2, 12)}
        full[10] = 16 * 6
        removed_sum = 0
        for v in range(2, 12):
            removed_sum += (full[v] - shoe.count(v)) * _HILO[v]
        assert removed_sum == running
        assert 52 <= len(shoe) <= 6 * 52


def test_deviation_learner_smoke():
    from blackjack.learn import MCDeviationLearner
    learner = MCDeviationLearner(seed=9)
    learner.train(5_000)
    assert learner.episodes == 5_000
    assert len(learner.index_report()) > 10   # header + 16 cells


def test_index_resolver_smoke():
    from blackjack.learn import IndexResolver, MCDeviationLearner
    learner = MCDeviationLearner(seed=11)
    learner.train(20_000)
    resolver = IndexResolver(learner)
    table = resolver.resolve_cell(16, 10, "S", pairs=200)
    assert table, "no buckets resolved"
    for b, (mean, n, se) in table.items():
        assert -2.0 <= mean <= 2.0 and n > 0 and se >= 0


def test_team_smoke():
    from blackjack.team import TeamSimulator
    team = TeamSimulator(Rules(), num_tables=3, num_bps=1,
                         call_in_tc=2.0, leave_tc=1.0, seed=4)
    res = team.run(3_000, bankroll=1e8, spotter_bet=25.0)
    assert res.ticks == 3_000
    assert res.spotter_rounds == 3 * 3_000
    assert 0.0 < res.bp_utilization < 1.0          # called in sometimes
    assert res.call_ins > 0 and res.bp_rounds > 0
    assert res.records[-1].bankroll == 1e8 + res.team_net


def test_team_leave_le_call_in():
    from blackjack.team import TeamSimulator
    try:
        TeamSimulator(Rules(), call_in_tc=1.0, leave_tc=2.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
