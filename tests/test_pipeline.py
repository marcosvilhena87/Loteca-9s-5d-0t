import tempfile
import unittest
from pathlib import Path

from scripts.common import Match, threshold_probability, ticket_metrics
from scripts.predict_results import optimize
from scripts.train_model import (exact_ticket, heuristic_ticket, probability_diagnostics,
                                 historical_ticket, position_rank_hit_rates,
                                 ticket_metrics_for, train, walk_forward_backtest)


class PipelineTests(unittest.TestCase):
    def test_tie_priority_is_one_then_two_then_x(self):
        game = Match(1, 1, "A", "B", {"1": .4, "X": .2, "2": .4})
        self.assertEqual(game.ranking, ["1", "2", "X"])

    def test_ticket_shape_and_flamengo_constraint(self):
        games = [Match(1, i + 1, "A", "B", {"1": .5, "X": .3, "2": .2}) for i in range(14)]
        games[13] = Match(1, 14, "A", "FLAMENGO/RJ", {"1": .8, "X": .15, "2": .05})
        ticket, _ = optimize(games, "uncertainty")
        self.assertEqual(sorted(map(len, ticket)), [1] * 9 + [2] * 5)
        self.assertIn("2", ticket[13])
        self.assertEqual(sum(map(len, ticket)), 19)

    def test_threshold_probability_extremes(self):
        self.assertEqual(threshold_probability([1.0] * 14), 1.0)
        self.assertEqual(threshold_probability([0.0] * 14), 0.0)

    def test_ticket_metrics_known_case(self):
        metrics = ticket_metrics([0.5] * 14)
        self.assertAlmostEqual(metrics["p14"], 1 / 2**14)
        self.assertAlmostEqual(metrics["p13"], 14 / 2**14)
        self.assertAlmostEqual(metrics["expected_hits"], 7.0)

    def test_exact_optimizer_is_never_worse_than_heuristics(self):
        games = [Match(1, i + 1, "A", "B", {
            "1": 0.34 + i / 1000, "X": 0.33, "2": 0.33 - i / 1000
        }) for i in range(14)]
        exact, _ = exact_ticket(games)
        exact_probability = ticket_metrics_for(games, exact)["p13_plus"]
        for policy in ("gain", "uncertainty", "margin", "ratio"):
            heuristic, _ = heuristic_ticket(games, policy)
            self.assertGreaterEqual(
                exact_probability, ticket_metrics_for(games, heuristic)["p13_plus"] - 1e-15
            )

    def test_training_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            first = train("data/concursos_anteriores.csv", str(path))
            second = train("data/concursos_anteriores.csv", str(path))
            self.assertEqual(first, second)

    def test_probability_diagnostics_for_perfect_forecasts(self):
        games = []
        for jogo in range(1, 15):
            actual = ("1", "X", "2")[(jogo - 1) % 3]
            probabilities = {result: float(result == actual) for result in ("1", "X", "2")}
            games.append(Match(1, jogo, "A", "B", probabilities, actual))
        diagnostics = probability_diagnostics({1: games})
        self.assertEqual(diagnostics["multiclass_brier"], 0.0)
        self.assertEqual(diagnostics["log_loss"], 0.0)
        self.assertEqual(diagnostics["ece"], 0.0)
        self.assertEqual(len(diagnostics["position_rank_hit_rates"]), 14)
        self.assertTrue(all(rates == [1.0, 0.0, 0.0]
                            for rates in diagnostics["position_rank_hit_rates"]))

    def test_probability_diagnostics_rejects_invalid_bin_count(self):
        with self.assertRaises(ValueError):
            probability_diagnostics({}, calibration_bins=1)

    def test_historical_policies_allocate_doubles_from_position_scores(self):
        games = [Match(2, i + 1, "A", "B", {"1": .5, "X": .3, "2": .2})
                 for i in range(14)]
        rates = [[.8 - i / 100, .1 + i / 100, .1] for i in range(14)]
        top1_ticket, _ = historical_ticket(games, "hist_top1", rates)
        top2_ticket, _ = historical_ticket(games, "hist_top2", rates)
        self.assertEqual([i for i, pick in enumerate(top1_ticket) if len(pick) == 2],
                         [9, 10, 11, 12, 13])
        self.assertEqual(top1_ticket, top2_ticket)
        self.assertEqual(sum(map(len, top1_ticket)), 19)

    def test_position_rates_use_only_supplied_contests(self):
        past = [Match(1, i + 1, "A", "B", {"1": .6, "X": .3, "2": .1}, "1")
                for i in range(14)]
        future = [Match(2, i + 1, "A", "B", {"1": .6, "X": .3, "2": .1}, "X")
                  for i in range(14)]
        self.assertEqual(position_rank_hit_rates([past]), [[.5, .25, .25]] * 14)
        self.assertEqual(position_rank_hit_rates([past, future]), [[.4, .4, .2]] * 14)

    def test_walk_forward_reports_only_out_of_sample_contests(self):
        contests = {}
        for concurso in range(1, 4):
            contests[concurso] = [Match(concurso, i + 1, "A", "B",
                {"1": .6, "X": .3, "2": .1}, "1") for i in range(14)]
        result = walk_forward_backtest(contests, minimum_history=2)
        self.assertTrue(all(values["hits"] == 14 for values in result.values()))
        self.assertTrue(all(values["p13_plus_empirical"] == 1.0 for values in result.values()))
        self.assertTrue(all(values["14"] == 1 and values["<=9"] == 0
                            for values in result.values()))

    def test_walk_forward_exports_contest_level_audit(self):
        contests = {}
        for concurso in range(1, 4):
            contests[concurso] = [Match(concurso, i + 1, "A", "B",
                {"1": .6, "X": .3, "2": .1}, "1") for i in range(14)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "backtest.csv"
            walk_forward_backtest(contests, minimum_history=2, output_path=output)
            rows = output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 8)  # header + seven strategies for one test contest
        self.assertIn("p13_plus_empirical;p12_plus_empirical;double_games;ticket", rows[0])
        self.assertTrue(all("T1=14|T2=5|T3=0" in row for row in rows[1:]))


if __name__ == "__main__":
    unittest.main()
