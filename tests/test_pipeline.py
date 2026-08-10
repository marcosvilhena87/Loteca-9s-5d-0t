import tempfile
import unittest
from pathlib import Path

from scripts.common import Match, threshold_probability, ticket_metrics
from scripts.predict_results import optimize, predict
from scripts.train_model import (exact_ticket, heuristic_ticket, probability_diagnostics,
                                 allocator_diagnostics,
                                 allocated_ticket, error_recovery_model, historical_ticket,
                                 position_rank_hit_rates, recovery_context, recovery_scores,
                                 select_second_mark,
                                 reliability_scores, ticket_metrics_for,
                                 top1_meta_features, top1_meta_score,
                                 top1_reliability_model, train, walk_forward_backtest,
                                 walk_forward_reliability, walk_forward_second_mark,
                                 nested_walk_forward_second_mark,
                                 walk_forward_top1_meta)


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

    def test_top2_probability_is_explicitly_equivalent_to_gain(self):
        games = [Match(1, i + 1, "A", "B", {
            "1": .50 + i / 1000, "X": .30 - i / 2000, "2": .20 - i / 2000
        }) for i in range(14)]
        gain, _ = heuristic_ticket(games, "gain")
        top2, _ = heuristic_ticket(games, "top2_probability")
        self.assertEqual(gain, top2)

    def test_allocator_diagnostics_are_paired_and_leak_free(self):
        contests = {
            concurso: [Match(concurso, i + 1, "A", "B",
                {"1": .50, "X": .30, "2": .20}, "1") for i in range(14)]
            for concurso in range(1, 4)
        }
        diagnostics = allocator_diagnostics(contests, minimum_history=2)
        self.assertTrue(diagnostics["no_future_information"])
        self.assertEqual(diagnostics["test_contests"], 1)
        self.assertEqual(diagnostics["overlap_mean_of_5"]["gain__top2_probability"], 5.0)
        comparison = diagnostics["pairwise"]["gain__top2_probability"]
        self.assertEqual((comparison["wins"], comparison["ties"], comparison["losses"]),
                         (0, 1, 0))

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
        self.assertEqual(len(rows), 9)  # header + eight strategies for one test contest
        self.assertIn("p13_plus_empirical;p12_plus_empirical;double_games;ticket", rows[0])
        self.assertTrue(all("T1=14|T2=5|T3=0" in row for row in rows[1:]))

    def test_reliability_scores_correct_top1_from_historical_context(self):
        past = [[Match(1, i + 1, "A", "B", {"1": .5, "X": .3, "2": .2}, "1")
                 for i in range(14)]]
        game = Match(2, 1, "A", "B", {"1": .5, "X": .3, "2": .2})
        scores = reliability_scores(game, top1_reliability_model(past))
        self.assertGreater(scores["top1_residual"], .5)
        self.assertGreater(scores["top1_lift"], .5)
        self.assertAlmostEqual(scores["top1_reliability"], 15 / 16)

    def test_disagreement_audit_is_walk_forward_and_well_formed(self):
        contests = {}
        for concurso in range(1, 4):
            contests[concurso] = [Match(concurso, i + 1, "A", "B",
                {"1": .55 if i % 2 else .45, "X": .30, "2": .15 if i % 2 else .25},
                "1" if (concurso + i) % 2 else "X") for i in range(14)]
        audit = walk_forward_reliability(contests, minimum_history=2)
        self.assertEqual(set(audit), {"top1_residual", "top1_lift", "top1_reliability"})
        for values in audit.values():
            self.assertEqual(values["cases"], values["baseline_wins"] +
                             values["historical_wins"] + values["neutral"])
            self.assertGreaterEqual(values["historical_win_rate"], 0.0)
            self.assertLessEqual(values["historical_win_rate"], 1.0)

    def test_prediction_exports_reliability_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            output_path = Path(directory) / "predictions.csv"
            train("data/concursos_anteriores.csv", str(model_path), backtest_path=None)
            rows = predict("data/proximo_concurso.csv", str(model_path),
                           str(output_path), verbose=False)
        self.assertEqual(len(rows), 14)
        self.assertTrue(all("top1_residual" in row and "top1_lift" in row and
                            "top1_reliability" in row for row in rows))
        self.assertTrue(all("p_top1_meta" in row and "top1_meta_delta" in row
                            for row in rows))

    def test_top1_meta_features_and_score_are_well_formed(self):
        game = Match(1, 1, "A", "B", {"1": .5, "X": .3, "2": .2})
        features = top1_meta_features(game)
        self.assertEqual(len(features), 10)
        self.assertEqual(features[0], 1.0)
        self.assertAlmostEqual(top1_meta_score(game, [0.0] * 10), .5)

    def test_top1_meta_walk_forward_has_no_future_observations(self):
        contests = {}
        for concurso in range(1, 4):
            contests[concurso] = [Match(concurso, i + 1, "A", "B",
                {"1": .6, "X": .3, "2": .1}, "1") for i in range(14)]
        audit = walk_forward_top1_meta(contests, minimum_history=2)
        self.assertEqual(audit["observations"], 14)
        self.assertFalse(audit["promoted_to_ticket"])
        self.assertEqual(audit["feature_names"][0], "intercept")

    def test_recovery_learns_top3_only_from_top1_misses(self):
        past = [[Match(1, i + 1, "A", "B", {"1": .5, "X": .3, "2": .2},
                       "2" if i < 10 else "1") for i in range(14)]]
        game = Match(2, 1, "A", "B", {"1": .5, "X": .3, "2": .2})
        model = error_recovery_model(past)
        scores = recovery_scores(game, model)
        self.assertGreater(scores["recovery_top3"], scores["recovery_top2"])
        self.assertAlmostEqual(scores["recovery_top2"] + scores["recovery_top3"], 1.0)
        self.assertEqual(select_second_mark(game, "recovery", model), "2")

    def test_allocator_and_second_mark_selector_preserve_ticket_shape(self):
        games = [Match(2, i + 1, "A", "B", {"1": .5, "X": .3, "2": .2})
                 for i in range(14)]
        context = recovery_context(games[0])
        recovery = {context: {"recovery_top2": .1, "recovery_top3": .9}}
        ticket, _ = allocated_ticket(games, "uncertainty", "recovery",
                                     recovery_model=recovery)
        self.assertEqual(sorted(map(len, ticket)), [1] * 9 + [2] * 5)
        self.assertTrue(all(pick == {"1", "2"} for pick in ticket if len(pick) == 2))

    def test_threshold_recovery_requires_sufficient_signed_advantage(self):
        game = Match(2, 1, "A", "B", {"1": .5, "X": .3, "2": .2})
        context = recovery_context(game)
        recovery = {context: {"recovery_top2": .46, "recovery_top3": .54}}
        self.assertEqual(select_second_mark(game, "threshold_recovery", recovery, .05), "2")
        self.assertEqual(select_second_mark(game, "threshold_recovery", recovery, .10), "X")

    def test_second_mark_audit_reports_thresholds_segments_and_bootstrap(self):
        contests = {}
        for concurso in range(1, 4):
            contests[concurso] = [Match(concurso, i + 1, "A", "B",
                {"1": .50, "X": .30, "2": .20}, "2") for i in range(14)]
        audit = walk_forward_second_mark(contests, minimum_history=2)
        self.assertEqual(set(audit["threshold_results"]),
                         {"0.00", "0.02", "0.05", "0.10", "0.15"})
        self.assertEqual(set(audit["by_gap_23"]),
                         {"0-2pp", "2-5pp", "5-10pp", "10pp+"})
        self.assertEqual(audit["bootstrap_resamples"], 2000)
        self.assertEqual(len(audit["recovery_win_rate_ci95"]), 2)
        self.assertEqual(audit["threshold_results"]["0.10"]["net_recovery_gain"], 14)

    def test_nested_threshold_selection_is_prospective_and_preserves_constraints(self):
        contests = {}
        for concurso in range(1, 5):
            contests[concurso] = [Match(concurso, i + 1, "A", "B",
                {"1": .50, "X": .30, "2": .20}, "2") for i in range(14)]
        audit = nested_walk_forward_second_mark(contests, minimum_history=2)
        self.assertTrue(audit["no_future_information"])
        self.assertEqual(audit["test_contests"], 2)
        self.assertEqual(audit["selections"][0]["past_observations"], 14)
        self.assertEqual(audit["selections"][1]["past_observations"], 28)
        self.assertEqual(sum(audit["threshold_usage"].values()), 2)
        self.assertGreaterEqual(audit["nested"]["mean"], audit["baseline"]["mean"])


if __name__ == "__main__":
    unittest.main()
