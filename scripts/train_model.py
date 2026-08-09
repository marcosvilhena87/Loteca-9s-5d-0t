"""Historical selection of the five-double allocation policy."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from itertools import combinations
from pathlib import Path

from scripts.common import Match, group_contests, normalize_team, read_matches, ticket_metrics

PROBABILITY_POLICIES = ("gain", "uncertainty", "margin", "ratio")
HISTORICAL_POLICIES = ("hist_top1", "hist_top2")
POLICIES = (*PROBABILITY_POLICIES, *HISTORICAL_POLICIES)
MIN_WALK_FORWARD_CONTESTS = 30
RELIABILITY_METRICS = ("top1_residual", "top1_lift", "top1_reliability")
P_TOP1_BINS = (0.40, 0.45, 0.50, 0.60)
MARGIN_BINS = (0.05, 0.10, 0.20)
META_FEATURE_NAMES = ("intercept", "p_top1", "p_top2", "p_top3",
                      "margin_top1_top2", "ratio_top2_top1", "entropy",
                      "top1_is_1", "top1_is_X", "top1_is_2")
DISAGREEMENT_STRENGTH_BINS = (0.02, 0.05, 0.10)
RECOVERY_SELECTORS = ("top2_baseline", "recovery", "threshold_recovery")
RECOVERY_THRESHOLDS = (0.00, 0.02, 0.05, 0.10, 0.15)
GAP_23_BINS = (0.02, 0.05, 0.10)


def _bin_index(value: float, boundaries: tuple[float, ...]) -> int:
    return sum(value >= boundary for boundary in boundaries)


def reliability_context(game: Match) -> tuple[int, int, str]:
    """Return the probability-only context used to calibrate Top1 confidence."""
    top1, top2, _ = game.ranking
    return (
        _bin_index(game.probabilities[top1], P_TOP1_BINS),
        _bin_index(game.probabilities[top1] - game.probabilities[top2], MARGIN_BINS),
        top1,
    )


def recovery_context(game: Match) -> tuple[int, int, str]:
    """Return a deliberately coarse, pre-match-only error-recovery context."""
    return reliability_context(game)


def error_recovery_model(
    contests: list[list[Match]],
) -> dict[tuple[int, int, str], dict[str, float]]:
    """Estimate which remaining rank wins when Top1 misses.

    Only Top1 misses enter the estimator. A symmetric Beta prior regularizes
    sparse contexts and guarantees complementary Top2/Top3 probabilities.
    """
    buckets: dict[tuple[int, int, str], list[int]] = {}
    for games in contests:
        for game in games:
            if game.actual not in ("1", "X", "2"):
                raise ValueError("recovery exige resultados reais")
            if game.actual == game.ranking[0]:
                continue
            counts = buckets.setdefault(recovery_context(game), [0, 0])
            counts[0] += 1
            counts[1] += game.actual == game.ranking[1]
    return {
        key: {
            "top1_misses": float(total),
            "recovery_top2": (top2_hits + 1) / (total + 2),
            "recovery_top3": (total - top2_hits + 1) / (total + 2),
        }
        for key, (total, top2_hits) in buckets.items()
    }


def recovery_scores(
    game: Match, model: dict[tuple[int, int, str], dict[str, float]],
) -> dict[str, float]:
    """Return smoothed recovery scores, using current odds as safe fallback."""
    bucket = model.get(recovery_context(game))
    if bucket is not None:
        return {name: bucket[name] for name in ("recovery_top2", "recovery_top3")}
    _, top2, top3 = game.ranking
    denominator = game.probabilities[top2] + game.probabilities[top3]
    return {
        "recovery_top2": game.probabilities[top2] / denominator,
        "recovery_top3": game.probabilities[top3] / denominator,
    }


def select_second_mark(game: Match, selector: str,
                       recovery_model: dict | None = None,
                       threshold: float = 0.0) -> str:
    """Select the protection mark independently from double allocation."""
    if selector == "top2_baseline":
        return game.ranking[1]
    if selector not in ("recovery", "threshold_recovery"):
        raise ValueError(f"seletor de segunda marca inválido: {selector}")
    if threshold < 0:
        raise ValueError("threshold de recovery não pode ser negativo")
    scores = recovery_scores(game, recovery_model or {})
    advantage = scores["recovery_top3"] - scores["recovery_top2"]
    return game.ranking[2] if advantage >= threshold and advantage > 0 else game.ranking[1]


def top1_reliability_model(contests: list[list[Match]]) -> dict[tuple[int, int, str], dict[str, float]]:
    """Learn leak-free, smoothed Top1 reliability statistics by context.

    A Beta(1, 1) prior prevents sparse contexts from producing confidence zero
    or one.  The mean forecast is tracked separately so residual and lift correct
    the baseline rather than replacing it.
    """
    buckets: dict[tuple[int, int, str], dict[str, float]] = {}
    for games in contests:
        for game in games:
            if game.actual not in ("1", "X", "2"):
                raise ValueError("confiabilidade exige resultados reais")
            key = reliability_context(game)
            bucket = buckets.setdefault(key, {"count": 0.0, "hits": 0.0, "p_sum": 0.0})
            bucket["count"] += 1
            bucket["hits"] += game.actual == game.ranking[0]
            bucket["p_sum"] += game.probabilities[game.ranking[0]]
    return {
        key: {
            "count": bucket["count"],
            "observed_rate": (bucket["hits"] + 1) / (bucket["count"] + 2),
            "mean_p_top1": bucket["p_sum"] / bucket["count"],
        }
        for key, bucket in buckets.items()
    }


def reliability_scores(game: Match, model: dict[tuple[int, int, str], dict[str, float]]) -> dict[str, float]:
    """Score Top1 with historical corrections, falling back safely to baseline."""
    p_top1 = game.probabilities[game.ranking[0]]
    bucket = model.get(reliability_context(game))
    if bucket is None:
        return {metric: p_top1 for metric in RELIABILITY_METRICS}
    observed = bucket["observed_rate"]
    mean_probability = bucket["mean_p_top1"]
    return {
        "top1_residual": min(1.0, max(0.0, p_top1 + observed - mean_probability)),
        "top1_lift": min(1.0, max(0.0, p_top1 * observed / mean_probability)),
        "top1_reliability": observed,
    }


def top1_meta_features(game: Match) -> list[float]:
    """Return the pre-match-only features specified by the README.

    Entropy is normalized by its three-outcome maximum so every continuous
    input is naturally close to [0, 1], which makes the small deterministic
    logistic learner stable without an external ML dependency.
    """
    top1, top2, top3 = game.ranking
    p1, p2, p3 = (game.probabilities[result] for result in game.ranking)
    entropy = -sum(p * math.log(max(p, 1e-15)) for p in (p1, p2, p3)) / math.log(3)
    return [1.0, p1, p2, p3, p1 - p2, p2 / p1, entropy,
            float(top1 == "1"), float(top1 == "X"), float(top1 == "2")]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def top1_meta_score(game: Match, coefficients: list[float]) -> float:
    if len(coefficients) != len(META_FEATURE_NAMES):
        raise ValueError("p(top1_meta) exige um coeficiente por feature")
    return _sigmoid(sum(weight * value for weight, value in
                        zip(coefficients, top1_meta_features(game))))


def fit_top1_meta(contests: list[list[Match]], epochs: int = 80,
                  learning_rate: float = 0.08, l2: float = 0.01,
                  initial: list[float] | None = None) -> list[float]:
    """Fit a deterministic regularized logistic model for ``top1_hit``."""
    coefficients = list(initial or [0.0] * len(META_FEATURE_NAMES))
    examples = [(top1_meta_features(game), float(game.actual == game.ranking[0]))
                for games in contests for game in games]
    if not examples:
        raise ValueError("p(top1_meta) exige histórico")
    for epoch in range(epochs):
        rate = learning_rate / (1.0 + epoch / 20.0)
        gradient = [0.0] * len(coefficients)
        for features, target in examples:
            error = _sigmoid(sum(w * x for w, x in zip(coefficients, features))) - target
            for index, value in enumerate(features):
                gradient[index] += error * value
        for index in range(len(coefficients)):
            penalty = 0.0 if index == 0 else l2 * coefficients[index]
            coefficients[index] -= rate * (gradient[index] / len(examples) + penalty)
    return coefficients


def _audit_summary(audit: dict[str, int]) -> dict[str, int | float]:
    informative = audit["baseline_wins"] + audit["meta_wins"]
    return {**audit, "meta_win_rate": round(audit["meta_wins"] / informative, 8)
            if informative else 0.0}


def walk_forward_top1_meta(contests: dict[int, list[Match]],
                           minimum_history: int = MIN_WALK_FORWARD_CONTESTS
                           ) -> dict[str, object]:
    """Evaluate p(Top1) against p(top1_meta) with an expanding cutoff.

    Each contest's predictions are made by a model fitted exclusively on older
    contests. Disagreement audits compare pair orderings, matching the existing
    reliability benchmark, and are segmented by correction strength and p(Top1).
    """
    ordered = [contests[key] for key in sorted(contests)]
    if not 1 <= minimum_history < len(ordered):
        raise ValueError("janela inicial inválida para walk-forward")
    total = {"cases": 0, "baseline_wins": 0, "meta_wins": 0, "neutral": 0}
    strength = {label: {"cases": 0, "baseline_wins": 0, "meta_wins": 0, "neutral": 0}
                for label in ("<0.02", "0.02-0.05", "0.05-0.10", ">=0.10")}
    probability = {label: {"cases": 0, "baseline_wins": 0, "meta_wins": 0, "neutral": 0}
                   for label in ("33-40%", "40-45%", "45-50%", "50-60%", "60%+")}
    baseline_brier = meta_brier = 0.0
    observations = 0
    coefficients = fit_top1_meta(ordered[:minimum_history])
    for index in range(minimum_history, len(ordered)):
        games = ordered[index]
        baseline = [game.probabilities[game.ranking[0]] for game in games]
        meta = [top1_meta_score(game, coefficients) for game in games]
        hits = [game.actual == game.ranking[0] for game in games]
        for base, candidate, hit in zip(baseline, meta, hits):
            baseline_brier += (base - hit) ** 2
            meta_brier += (candidate - hit) ** 2
            observations += 1
        for left, right in combinations(range(14), 2):
            base_order = (baseline[left] > baseline[right]) - (baseline[left] < baseline[right])
            meta_order = (meta[left] > meta[right]) - (meta[left] < meta[right])
            if not base_order or not meta_order or base_order == meta_order:
                continue
            result_order = int(hits[left]) - int(hits[right])
            delta = max(abs(meta[left] - baseline[left]), abs(meta[right] - baseline[right]))
            strength_label = ("<0.02" if delta < .02 else "0.02-0.05" if delta < .05
                              else "0.05-0.10" if delta < .10 else ">=0.10")
            mean_p = (baseline[left] + baseline[right]) / 2
            probability_label = ("33-40%" if mean_p < .40 else "40-45%" if mean_p < .45
                                 else "45-50%" if mean_p < .50 else "50-60%"
                                 if mean_p < .60 else "60%+")
            for audit in (total, strength[strength_label], probability[probability_label]):
                audit["cases"] += 1
                if not result_order:
                    audit["neutral"] += 1
                elif result_order == meta_order:
                    audit["meta_wins"] += 1
                else:
                    audit["baseline_wins"] += 1
        # Online updates preserve the expanding temporal cutoff without repeatedly
        # refitting thousands of already-seen examples at every contest.
        coefficients = fit_top1_meta([games], epochs=4, learning_rate=.03,
                                     initial=coefficients)
    return {
        "feature_names": list(META_FEATURE_NAMES),
        "coefficients": fit_top1_meta(ordered),
        "observations": observations,
        "baseline_brier": round(baseline_brier / observations, 8),
        "meta_brier": round(meta_brier / observations, 8),
        "disagreement": _audit_summary(total),
        "disagreement_by_strength": {key: _audit_summary(value)
                                      for key, value in strength.items()},
        "disagreement_by_p_top1": {key: _audit_summary(value)
                                    for key, value in probability.items()},
        "promoted_to_ticket": False,
    }


def walk_forward_reliability(contests: dict[int, list[Match]],
                             minimum_history: int = MIN_WALK_FORWARD_CONTESTS
                             ) -> dict[str, dict[str, int | float]]:
    """Run the README disagreement test using only earlier contests.

    Only informative pairs (exactly one Top1 hit) count as a win. Pairs tied by
    either ordering are neutral, making the comparison explicit and reproducible.
    """
    ordered = [contests[key] for key in sorted(contests)]
    if not 1 <= minimum_history < len(ordered):
        raise ValueError("janela inicial inválida para walk-forward")
    totals = {metric: {"cases": 0, "baseline_wins": 0, "historical_wins": 0,
                       "neutral": 0} for metric in RELIABILITY_METRICS}
    for index in range(minimum_history, len(ordered)):
        model = top1_reliability_model(ordered[:index])
        games = ordered[index]
        baseline = [game.probabilities[game.ranking[0]] for game in games]
        scores = [reliability_scores(game, model) for game in games]
        hits = [game.actual == game.ranking[0] for game in games]
        for metric in RELIABILITY_METRICS:
            for left, right in combinations(range(len(games)), 2):
                baseline_order = (baseline[left] > baseline[right]) - (baseline[left] < baseline[right])
                historical_order = ((scores[left][metric] > scores[right][metric]) -
                                    (scores[left][metric] < scores[right][metric]))
                if not baseline_order or not historical_order or baseline_order == historical_order:
                    continue
                result_order = int(hits[left]) - int(hits[right])
                audit = totals[metric]
                audit["cases"] += 1
                if not result_order:
                    audit["neutral"] += 1
                elif result_order == historical_order:
                    audit["historical_wins"] += 1
                else:
                    audit["baseline_wins"] += 1
    for audit in totals.values():
        informative = audit["historical_wins"] + audit["baseline_wins"]
        audit["historical_win_rate"] = round(
            audit["historical_wins"] / informative, 8
        ) if informative else 0.0
    return totals


def walk_forward_second_mark(contests: dict[int, list[Match]],
                             minimum_history: int = MIN_WALK_FORWARD_CONTESTS
                             ) -> dict[str, object]:
    """Audit Top2 versus recovery exclusively on out-of-sample Top1 misses."""
    ordered = [contests[key] for key in sorted(contests)]
    if not 1 <= minimum_history < len(ordered):
        raise ValueError("janela inicial inválida para walk-forward")
    observations: list[dict[str, object]] = []
    for index in range(minimum_history, len(ordered)):
        model = error_recovery_model(ordered[:index])
        for game in ordered[index]:
            if game.actual == game.ranking[0]:
                continue
            scores = recovery_scores(game, model)
            observations.append({
                "advantage": scores["recovery_top3"] - scores["recovery_top2"],
                "gap_23": game.probabilities[game.ranking[1]] - game.probabilities[game.ranking[2]],
                "p_top1": game.probabilities[game.ranking[0]],
                "top3_hit": game.actual == game.ranking[2],
            })

    def summarize(rows: list[dict[str, object]], threshold: float = 0.0) -> dict[str, object]:
        switched = [row for row in rows if float(row["advantage"]) >= threshold
                    and float(row["advantage"]) > 0]
        recovery_wins = sum(bool(row["top3_hit"]) for row in switched)
        top2_wins = len(switched) - recovery_wins
        rate = recovery_wins / len(switched) if switched else 0.0
        # Fixed seed makes the 2,000-resample percentile interval reproducible.
        rng = random.Random(20250613 + round(threshold * 100))
        samples = []
        outcomes = [int(bool(row["top3_hit"])) for row in switched]
        if outcomes:
            for _ in range(2000):
                samples.append(sum(rng.choice(outcomes) for _ in outcomes) / len(outcomes))
            samples.sort()
            ci = [samples[49], samples[1949]]
        else:
            ci = [0.0, 0.0]
        return {
            "cases": len(switched), "top2_baseline_wins": top2_wins,
            "recovery_wins": recovery_wins, "neutral": 0,
            "recovery_win_rate": round(rate, 8),
            "recovery_win_rate_ci95": [round(value, 8) for value in ci],
            "bootstrap_resamples": 2000,
            "statistically_distinguishable_from_50pct": bool(ci[0] > .5 or ci[1] < .5),
        }

    threshold_results = {f"{threshold:.2f}": summarize(observations, threshold)
                         for threshold in RECOVERY_THRESHOLDS}
    audit = threshold_results["0.00"]
    gap_labels = ("0-2pp", "2-5pp", "5-10pp", "10pp+")
    p_labels = ("33-40%", "40-45%", "45-50%", "50-60%", "60%+")
    by_gap = {label: summarize([row for row in observations
                               if _bin_index(float(row["gap_23"]), GAP_23_BINS) == index])
              for index, label in enumerate(gap_labels)}
    by_p_top1 = {label: summarize([row for row in observations
                                  if _bin_index(float(row["p_top1"]), P_TOP1_BINS) == index])
                 for index, label in enumerate(p_labels)}
    return {
        **audit,
        "top1_misses": len(observations), "recovery_top3_uses": audit["cases"],
        "recovery_top3_usage_rate": round(audit["cases"] / len(observations), 8)
        if observations else 0.0,
        "threshold_results": threshold_results,
        "by_gap_23": by_gap, "by_p_top1": by_p_top1,
        "passes_disagreement_threshold": bool(audit["cases"] and
                                              audit["recovery_wins"] > audit["top2_baseline_wins"]),
        # Promotion additionally requires ticket-level P13+/P12+ evidence.
        "promoted_to_ticket": False,
    }


def probability_diagnostics(contests: dict[int, list[Match]],
                            calibration_bins: int = 10) -> dict[str, object]:
    """Measure probability quality without using any post-result information.

    Brier and log loss evaluate all three outcomes.  ECE groups the 3*N outcome
    probabilities into equal-width bins; empty bins are omitted from the audit
    trail.  The position matrix is deliberately descriptive: it is not fed back
    into ticket generation before a walk-forward validation exists.
    """
    if calibration_bins < 2:
        raise ValueError("calibration_bins deve ser pelo menos 2")
    games = [game for contest in contests.values() for game in contest]
    if not games or any(game.actual not in ("1", "X", "2") for game in games):
        raise ValueError("diagnóstico exige resultados históricos válidos")

    epsilon = 1e-15
    brier = 0.0
    log_loss = 0.0
    bins = [{"count": 0, "probability_sum": 0.0, "hits": 0}
            for _ in range(calibration_bins)]
    position_counts = [[0, 0, 0] for _ in range(14)]
    position_totals = [0] * 14
    for game in games:
        log_loss -= math.log(max(game.probabilities[game.actual], epsilon))
        for result, probability in game.probabilities.items():
            observed = int(result == game.actual)
            brier += (probability - observed) ** 2
            index = min(int(probability * calibration_bins), calibration_bins - 1)
            bins[index]["count"] += 1
            bins[index]["probability_sum"] += probability
            bins[index]["hits"] += observed
        position_counts[game.jogo - 1][game.ranking.index(game.actual)] += 1
        position_totals[game.jogo - 1] += 1

    calibration = []
    ece = 0.0
    observations = len(games) * 3
    for index, bucket in enumerate(bins):
        if not bucket["count"]:
            continue
        mean_probability = bucket["probability_sum"] / bucket["count"]
        observed_rate = bucket["hits"] / bucket["count"]
        ece += bucket["count"] / observations * abs(mean_probability - observed_rate)
        calibration.append({
            "lower": round(index / calibration_bins, 6),
            "upper": round((index + 1) / calibration_bins, 6),
            "count": bucket["count"],
            "mean_probability": round(mean_probability, 6),
            "observed_rate": round(observed_rate, 6),
        })
    position_rank_hit_rates = [
        [round(count / total, 6) for count in counts]
        for counts, total in zip(position_counts, position_totals)
    ]
    return {
        "multiclass_brier": round(brier / len(games), 6),
        "log_loss": round(log_loss / len(games), 6),
        "ece": round(ece, 6),
        "calibration_bins": calibration,
        "position_rank_hit_rates": position_rank_hit_rates,
    }


def priority(match: Match, policy: str) -> float:
    first, second, _ = match.ranking
    p1, p2 = match.probabilities[first], match.probabilities[second]
    return {
        "gain": p2,
        "uncertainty": 1.0 - p1,
        "margin": 1.0 - (p1 - p2),
        "ratio": p2 / p1,
    }[policy]


def team_result(match: Match, needle: str) -> str | None:
    home, away = normalize_team(match.mandante), normalize_team(match.visitante)
    return "1" if needle in home else ("2" if needle in away else None)


def constrained_pick(game: Match, is_double: bool,
                     palmeiras_threshold: float,
                     second_mark: str | None = None) -> tuple[set[str], list[str]]:
    """Create one constrained pick without changing its number of markings."""
    if second_mark is not None and (not is_double or second_mark == game.ranking[0]
                                    or second_mark not in ("1", "X", "2")):
        raise ValueError("segunda marca exige um duplo e deve ser diferente do Top1")
    selection = ({game.ranking[0], second_mark} if second_mark is not None
                 else set(game.ranking[:2] if is_double else game.ranking[:1]))
    notes: list[str] = []
    flamengo_win = team_result(game, "FLAMENGO")
    if flamengo_win and flamengo_win not in selection:
        removed = min(selection, key=lambda result: game.probabilities[result])
        selection.remove(removed)
        selection.add(flamengo_win)
    if flamengo_win:
        notes.append(f"FLAMENGO jogo {game.jogo}: vitória {flamengo_win} coberta")

    palmeiras_win = team_result(game, "PALMEIRAS")
    if palmeiras_win in selection:
        replacement = next(result for result in game.ranking if result != palmeiras_win)
        loss = game.probabilities[palmeiras_win] - game.probabilities[replacement]
        if loss <= palmeiras_threshold and not is_double:
            selection = {replacement}
            notes.append(f"PALMEIRAS jogo {game.jogo}: vitória excluída (perda {loss:.3f})")
        else:
            notes.append(f"PALMEIRAS jogo {game.jogo}: preferência não aplicada (perda {loss:.3f})")
    return selection, notes


def build_ticket(games: list[Match], double_indexes: set[int],
                 palmeiras_threshold: float = 0.03,
                 second_marks: dict[int, str] | None = None
                 ) -> tuple[list[set[str]], list[str]]:
    """Build every ticket through the same constraints-aware pipeline."""
    if len(games) != 14 or len(double_indexes) != 5 or not double_indexes <= set(range(14)):
        raise ValueError("o ticket exige 14 jogos e exatamente 5 índices de duplos")
    if second_marks and set(second_marks) != double_indexes:
        raise ValueError("segunda marca deve ser informada para cada duplo")
    ticket: list[set[str]] = []
    notes: list[str] = []
    for i, game in enumerate(games):
        selection, pick_notes = constrained_pick(
            game, i in double_indexes, palmeiras_threshold,
            second_marks.get(i) if second_marks else None,
        )
        ticket.append(selection)
        notes.extend(pick_notes)
    return ticket, notes


def ticket_metrics_for(games: list[Match], ticket: list[set[str]]) -> dict[str, float]:
    covered = [sum(game.probabilities[result] for result in pick)
               for game, pick in zip(games, ticket)]
    return ticket_metrics(covered)


def heuristic_ticket(games: list[Match], policy: str) -> tuple[list[set[str]], list[str]]:
    indexes = set(sorted(range(14), key=lambda i: (-priority(games[i], policy), i))[:5])
    return build_ticket(games, indexes)


def allocated_ticket(games: list[Match], policy: str, selector: str,
                     rates: list[list[float]] | None = None,
                     recovery_model: dict | None = None) -> tuple[list[set[str]], list[str]]:
    """Compose a DoubleAllocator and a SecondMarkSelector into one valid ticket."""
    baseline, _ = ticket_for_policy(games, policy, rates)
    indexes = {i for i, pick in enumerate(baseline) if len(pick) == 2}
    marks = {i: select_second_mark(games[i], selector, recovery_model) for i in indexes}
    return build_ticket(games, indexes, second_marks=marks)


def position_rank_hit_rates(contests: list[list[Match]]) -> list[list[float]]:
    """Estimate position/rank rates using only the supplied past contests.

    A Dirichlet(1, 1, 1) prior keeps the ordering stable when the walk-forward
    window is still small and prevents zero-frequency conclusions.
    """
    counts = [[1, 1, 1] for _ in range(14)]
    for games in contests:
        for game in games:
            if game.actual not in ("1", "X", "2"):
                raise ValueError("score histórico exige resultados reais")
            counts[game.jogo - 1][game.ranking.index(game.actual)] += 1
    return [[count / sum(row) for count in row] for row in counts]


def historical_ticket(games: list[Match], policy: str,
                      rates: list[list[float]]) -> tuple[list[set[str]], list[str]]:
    """Allocate doubles from historical position evidence.

    ``hist_top1`` protects the five positions where Top1 has been least reliable;
    ``hist_top2`` covers the five positions where Top2 has hit most often.
    Results used to estimate ``rates`` are intentionally supplied by the caller,
    allowing the trainer to enforce a leak-free walk-forward cutoff.
    """
    if policy not in HISTORICAL_POLICIES:
        raise ValueError(f"política histórica inválida: {policy}")
    if len(rates) != 14 or any(len(row) != 3 for row in rates):
        raise ValueError("score histórico exige uma matriz 14 x 3")
    rank_index = 0 if policy == "hist_top1" else 1
    indexes = set(sorted(
        range(14),
        key=lambda i: ((rates[i][rank_index] if policy == "hist_top1"
                        else -rates[i][rank_index]), i),
    )[:5])
    return build_ticket(games, indexes)


def ticket_for_policy(games: list[Match], policy: str,
                      rates: list[list[float]] | None = None) -> tuple[list[set[str]], list[str]]:
    if policy == "exact":
        return exact_ticket(games)
    if policy in HISTORICAL_POLICIES:
        if rates is None:
            raise ValueError("política histórica exige scores por posição")
        return historical_ticket(games, policy, rates)
    return heuristic_ticket(games, policy)


def walk_forward_backtest(contests: dict[int, list[Match]],
                          minimum_history: int = MIN_WALK_FORWARD_CONTESTS,
                          output_path: str | Path | None = None
                          ) -> dict[str, dict[str, int | float]]:
    """Compare policies prospectively and optionally persist contest-level evidence.

    The exported empirical rates are cumulative within each strategy and therefore
    use only evaluations observed up to that row.  They must not be confused with
    the modelled Poisson-binomial probability of a particular ticket.
    """
    contest_ids = sorted(contests)
    ordered = [contests[key] for key in contest_ids]
    if not 1 <= minimum_history < len(ordered):
        raise ValueError("janela inicial inválida para walk-forward")
    strategies = (*POLICIES, "exact")
    hit_history: dict[str, list[int]] = {policy: [] for policy in strategies}
    records: list[dict[str, object]] = []
    for index in range(minimum_history, len(ordered)):
        rates = position_rank_hit_rates(ordered[:index])
        games = ordered[index]
        for policy in strategies:
            ticket, _ = ticket_for_policy(games, policy, rates)
            hits = sum(game.actual in selection
                       for game, selection in zip(games, ticket))
            history = hit_history[policy]
            history.append(hits)
            double_games = [game.jogo for game, pick in zip(games, ticket) if len(pick) == 2]
            records.append({
                "concurso": contest_ids[index], "strategy": policy,
                "ordering": policy, "distribution_id": "T1=14|T2=5|T3=0",
                "hits": hits, "hit_14": int(hits == 14), "hit_13": int(hits == 13),
                "hit_12": int(hits == 12),
                "p13_plus_empirical": f"{sum(value >= 13 for value in history) / len(history):.8f}",
                "p12_plus_empirical": f"{sum(value >= 12 for value in history) / len(history):.8f}",
                "double_games": ",".join(map(str, double_games)),
                "ticket": "|".join("".join(r for r in ("1", "X", "2") if r in pick)
                                     for pick in ticket),
            })
    evaluations: dict[str, dict[str, int | float]] = {}
    for policy, hits in hit_history.items():
        evaluations[policy] = {
            "14": sum(value == 14 for value in hits),
            "13": sum(value == 13 for value in hits),
            "12": sum(value == 12 for value in hits),
            "11": sum(value == 11 for value in hits),
            "10": sum(value == 10 for value in hits),
            "<=9": sum(value <= 9 for value in hits),
            "hits": sum(hits), "mean": round(statistics.fmean(hits), 6),
            "median": float(statistics.median(hits)),
            "stddev": round(statistics.pstdev(hits), 6),
            "min": min(hits), "max": max(hits),
            "p13_plus_empirical": round(sum(value >= 13 for value in hits) / len(hits), 8),
            "p12_plus_empirical": round(sum(value >= 12 for value in hits) / len(hits), 8),
        }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys(), delimiter=";",
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    return evaluations


def exact_ticket(games: list[Match]) -> tuple[list[set[str]], list[str]]:
    """Evaluate all C(14, 5)=2,002 allocations and maximize P(>=13)."""
    if len(games) != 14:
        raise ValueError("o ticket exige exatamente 14 jogos")
    # Constraints depend only on whether a match is dry or double. Precomputing
    # both alternatives avoids rebuilding sets and normalized team names 2,002 times.
    single_coverage: list[float] = []
    double_coverage: list[float] = []
    for game in games:
        single, _ = constrained_pick(game, False, 0.03)
        double, _ = constrained_pick(game, True, 0.03)
        single_coverage.append(sum(game.probabilities[r] for r in single))
        double_coverage.append(sum(game.probabilities[r] for r in double))
    best_indexes: tuple[int, ...] | None = None
    best_key: tuple[float, float, float, tuple[int, ...]] | None = None
    for indexes_tuple in combinations(range(14), 5):
        index_set = set(indexes_tuple)
        covered = [double_coverage[i] if i in index_set else single_coverage[i]
                   for i in range(14)]
        metrics = ticket_metrics(covered)
        # Deterministic final tie-break favours earlier games (Top1 concentration).
        key = (metrics["p13_plus"], metrics["p14"], metrics["expected_hits"],
               tuple(-index for index in indexes_tuple))
        if best_key is None or key > best_key:
            best_key, best_indexes = key, indexes_tuple
    assert best_indexes is not None
    return build_ticket(games, set(best_indexes))


def train(history_path: str, model_path: str,
          backtest_path: str | Path | None = "output/backtest.csv") -> dict[str, object]:
    contests = group_contests(read_matches(history_path, require_actual=True))
    evaluations = walk_forward_backtest(contests, output_path=backtest_path)
    # Goal hierarchy: 14, 13+, 12+, stability and finally average hits.
    strategies = (*POLICIES, "exact")
    selected = max(strategies, key=lambda p: (
        evaluations[p]["14"],
        evaluations[p]["13"] + evaluations[p]["14"],
        evaluations[p]["12"] + evaluations[p]["13"] + evaluations[p]["14"],
        -evaluations[p]["stddev"], evaluations[p]["mean"], -strategies.index(p),
    ))
    rank_hits = [0, 0, 0]
    for games in contests.values():
        for game in games:
            rank_hits[game.ranking.index(game.actual)] += 1
    top1_meta = walk_forward_top1_meta(contests)
    recovery = error_recovery_model(list(contests.values()))
    recovery_audit = walk_forward_second_mark(contests)
    model = {
        "version": 9, "selected_policy": selected,
        "selected_second_mark": "top2_baseline",
        "contests_evaluated": len(contests), "policy_backtest": evaluations,
        "walk_forward": {
            "minimum_history": MIN_WALK_FORWARD_CONTESTS,
            "test_contests": len(contests) - MIN_WALK_FORWARD_CONTESTS,
            "no_future_information": True,
        },
        "position_rank_hit_rates": position_rank_hit_rates(list(contests.values())),
        "rank_hit_rates": [round(count / sum(rank_hits), 6) for count in rank_hits],
        "probability_diagnostics": probability_diagnostics(contests),
        "top1_meta": top1_meta,
        "top1_reliability": {
            "walk_forward_disagreement": walk_forward_reliability(contests),
            "contexts": [
                {"p_top1_bin": key[0], "margin_bin": key[1], "top1_result": key[2],
                 **{name: round(value, 8) for name, value in values.items()}}
                for key, values in sorted(top1_reliability_model(list(contests.values())).items())
            ],
        },
        "error_recovery": {
            "walk_forward_disagreement": recovery_audit,
            "contexts": [
                {"p_top1_bin": key[0], "margin_bin": key[1], "top1_result": key[2],
                 **{name: round(value, 8) for name, value in values.items()}}
                for key, values in sorted(recovery.items())
            ],
        },
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return model
