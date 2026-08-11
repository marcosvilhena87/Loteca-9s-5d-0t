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

PROBABILITY_POLICIES = ("gain", "top2_probability", "uncertainty", "margin", "ratio")
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
SAFE_DISTRIBUTIONS = tuple((14, top2, 5 - top2) for top2 in range(5, -1, -1))


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
            "net_recovery_gain": recovery_wins - top2_wins,
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


def nested_walk_forward_second_mark(
    contests: dict[int, list[Match]],
    minimum_history: int = MIN_WALK_FORWARD_CONTESTS,
    allocator: str = "uncertainty",
) -> dict[str, object]:
    """Select a recovery threshold using only evidence available before each test.

    Inner observations are themselves out of sample: a historical game's recovery
    score is produced from contests older than that game.  The chosen threshold is
    then frozen for the next contest.  A deterministic, conservative tie-break
    prefers fewer switches and the larger threshold.
    """
    contest_ids = sorted(contests)
    ordered = [contests[key] for key in contest_ids]
    if not 1 <= minimum_history < len(ordered):
        raise ValueError("janela inicial inválida para nested walk-forward")

    inner_observations: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    baseline_hits: list[int] = []
    nested_hits: list[int] = []
    threshold_usage = {f"{value:.2f}": 0 for value in RECOVERY_THRESHOLDS}

    def append_out_of_sample_observations(index: int) -> None:
        model = error_recovery_model(ordered[:index])
        for game in ordered[index]:
            if game.actual == game.ranking[0]:
                continue
            scores = recovery_scores(game, model)
            inner_observations.append({
                "advantage": scores["recovery_top3"] - scores["recovery_top2"],
                "top3_hit": game.actual == game.ranking[2],
            })

    def threshold_score(threshold: float) -> tuple[int, int, float]:
        switched = [row for row in inner_observations
                    if float(row["advantage"]) >= threshold
                    and float(row["advantage"]) > 0]
        wins = sum(bool(row["top3_hit"]) for row in switched)
        net_gain = 2 * wins - len(switched)
        # Net gain is primary; fewer switches and a larger threshold guard ties.
        return net_gain, -len(switched), threshold

    # Seed the inner audit from the initial window while retaining a temporal
    # cutoff for every score (contest 2 uses contest 1, and so on).
    for index in range(1, minimum_history):
        append_out_of_sample_observations(index)

    for index in range(minimum_history, len(ordered)):
        threshold = max(RECOVERY_THRESHOLDS, key=threshold_score)
        threshold_key = f"{threshold:.2f}"
        threshold_usage[threshold_key] += 1
        games = ordered[index]
        rates = position_rank_hit_rates(ordered[:index])
        recovery_model = error_recovery_model(ordered[:index])
        baseline, _ = allocated_ticket(games, allocator, "top2_baseline", rates,
                                       recovery_model)
        nested, _ = allocated_ticket(games, allocator, "threshold_recovery", rates,
                                     recovery_model, threshold)
        base_hits = sum(game.actual in pick for game, pick in zip(games, baseline))
        candidate_hits = sum(game.actual in pick for game, pick in zip(games, nested))
        baseline_hits.append(base_hits)
        nested_hits.append(candidate_hits)
        selections.append({
            "concurso": contest_ids[index], "threshold": threshold,
            "past_observations": len(inner_observations),
            "baseline_hits": base_hits, "nested_hits": candidate_hits,
        })

        # Add the just-tested contest only after its frozen decision is recorded.
        append_out_of_sample_observations(index)

    def ticket_summary(hits: list[int]) -> dict[str, int | float]:
        return {
            "14": sum(value == 14 for value in hits),
            "13": sum(value == 13 for value in hits),
            "12": sum(value == 12 for value in hits),
            "p13_plus_empirical": round(sum(value >= 13 for value in hits) / len(hits), 8),
            "p12_plus_empirical": round(sum(value >= 12 for value in hits) / len(hits), 8),
            "mean": round(statistics.fmean(hits), 6),
        }

    return {
        "allocator": allocator, "test_contests": len(nested_hits),
        "no_future_information": True, "threshold_usage": threshold_usage,
        "selections": selections,
        "baseline": ticket_summary(baseline_hits),
        "nested": ticket_summary(nested_hits),
        "delta_p13_plus": round(
            ticket_summary(nested_hits)["p13_plus_empirical"]
            - ticket_summary(baseline_hits)["p13_plus_empirical"], 8),
        "delta_p12_plus": round(
            ticket_summary(nested_hits)["p12_plus_empirical"]
            - ticket_summary(baseline_hits)["p12_plus_empirical"], 8),
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
        # Keep this nominal baseline separate from ``gain``.  Gain may evolve
        # into a conditional value estimate, while this policy must always mean
        # the five largest raw p(Top2) values.
        "top2_probability": p2,
        "uncertainty": 1.0 - p1,
        "margin": 1.0 - (p1 - p2),
        "ratio": p2 / p1,
    }[policy]


def allocator_diagnostics(contests: dict[int, list[Match]],
                          minimum_history: int = MIN_WALK_FORWARD_CONTESTS
                          ) -> dict[str, object]:
    """Compare allocators contest by contest without using future outcomes.

    Overlap exposes policies that merely rename the same five doubles. Pairwise
    results retain the dependence between tickets and separately compare the
    rare target event (13+), avoiding conclusions based only on mean accuracy.
    """
    contest_ids = sorted(contests)
    ordered = [contests[key] for key in contest_ids]
    if not 1 <= minimum_history < len(ordered):
        raise ValueError("janela inicial inválida para diagnóstico de allocators")
    strategies = (*POLICIES, "exact")
    indexes_by_policy = {policy: [] for policy in strategies}
    hits_by_policy = {policy: [] for policy in strategies}
    for index in range(minimum_history, len(ordered)):
        assert contest_ids[index - 1] < contest_ids[index]
        rates = position_rank_hit_rates(ordered[:index])
        games = ordered[index]
        for policy in strategies:
            ticket, _ = ticket_for_policy(games, policy, rates)
            indexes_by_policy[policy].append(
                {i for i, pick in enumerate(ticket) if len(pick) == 2}
            )
            hits_by_policy[policy].append(sum(
                game.actual in pick for game, pick in zip(games, ticket)
            ))
    overlap: dict[str, float] = {}
    pairwise: dict[str, dict[str, int | float]] = {}
    for left, right in combinations(strategies, 2):
        key = f"{left}__{right}"
        left_hits, right_hits = hits_by_policy[left], hits_by_policy[right]
        overlap[key] = round(statistics.fmean(
            len(a & b) for a, b in zip(indexes_by_policy[left], indexes_by_policy[right])
        ), 6)
        deltas = [a - b for a, b in zip(left_hits, right_hits)]
        left_tail = [value >= 13 for value in left_hits]
        right_tail = [value >= 13 for value in right_hits]
        pairwise[key] = {
            "wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
            "mean_delta_hits": round(statistics.fmean(deltas), 8),
            "p13_plus_wins": sum(a and not b for a, b in zip(left_tail, right_tail)),
            "p13_plus_ties": sum(a == b for a, b in zip(left_tail, right_tail)),
            "p13_plus_losses": sum(b and not a for a, b in zip(left_tail, right_tail)),
        }
    return {
        "test_contests": len(ordered) - minimum_history,
        "no_future_information": True,
        "overlap_mean_of_5": overlap,
        "pairwise": pairwise,
    }


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
                     recovery_model: dict | None = None,
                     recovery_threshold: float = 0.0) -> tuple[list[set[str]], list[str]]:
    """Compose a DoubleAllocator and a SecondMarkSelector into one valid ticket."""
    baseline, _ = ticket_for_policy(games, policy, rates)
    indexes = {i for i, pick in enumerate(baseline) if len(pick) == 2}
    marks = {i: select_second_mark(games[i], selector, recovery_model, recovery_threshold)
             for i in indexes}
    return build_ticket(games, indexes, second_marks=marks)


def _ticket_hits(games: list[Match], ticket: list[set[str]]) -> int:
    """Score a historical ticket; deliberately unusable without outcomes."""
    if any(game.actual not in ("1", "X", "2") for game in games):
        raise ValueError("oráculos exigem resultados reais")
    return sum(game.actual in pick for game, pick in zip(games, ticket))


def is_xyz_distribution_valid(x: int, y: int, z: int) -> bool:
    """Return whether rank totals can form a 9-dry/5-double ticket.

    There are 19 marks and a rank can occur at most once in each of the 14
    matches.  Those conditions are also sufficient: the five extra marks can
    always be paired with a different rank when no rank exceeds 14.
    """
    return all(isinstance(value, int) and 0 <= value <= 14 for value in (x, y, z)) \
        and x + y + z == 19


def generate_xyz_neighbors(x: int, y: int, z: int) -> list[tuple[int, int, int]]:
    """Generate every valid distribution one unit-transfer away."""
    if not is_xyz_distribution_valid(x, y, z):
        raise ValueError("a distribuição XYZ de origem é inválida")
    values = (x, y, z)
    neighbors = set()
    for source in range(3):
        for destination in range(3):
            if source == destination:
                continue
            candidate = list(values)
            candidate[source] -= 1
            candidate[destination] += 1
            if is_xyz_distribution_valid(*candidate):
                neighbors.add(tuple(candidate))
    return sorted(neighbors, reverse=True)


def generate_xyz_radius(center: tuple[int, int, int] = (9, 5, 5),
                        radius: int = 1) -> list[tuple[int, int, int]]:
    """Return the unique feasible XYZ ball up to ``radius`` transfers."""
    if not is_xyz_distribution_valid(*center) or not isinstance(radius, int) or radius < 0:
        raise ValueError("centro ou raio XYZ inválido")
    distances = {center: 0}
    frontier = {center}
    for distance in range(1, radius + 1):
        next_frontier = {
            neighbor for point in frontier for neighbor in generate_xyz_neighbors(*point)
            if neighbor not in distances
        }
        for point in next_frontier:
            distances[point] = distance
        frontier = next_frontier
    return sorted(distances, key=lambda point: (distances[point], tuple(-v for v in point)))


def xyz_distribution_id(distribution: tuple[int, int, int]) -> str:
    if not is_xyz_distribution_valid(*distribution):
        raise ValueError("distribuição XYZ inválida")
    return "XYZ_{:02d}_{:02d}_{:02d}".format(*distribution)


def xyz_distribution_ticket(games: list[Match], distribution: tuple[int, int, int]
                            ) -> tuple[list[set[str]], list[str]]:
    """Place a fixed XYZ composition using pre-match probabilities only.

    The dynamic program considers T1, T2, T3, T1T2, T1T3 and T2T3 per game,
    maximizing total covered probability while tracking exact global counts.
    Flamengo's victory is filtered into every candidate state, rather than
    repaired afterwards (which could silently alter the requested XYZ totals).
    """
    return _xyz_distribution_tickets(games, (distribution,))[distribution]


def _xyz_distribution_tickets(
    games: list[Match], distributions: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]],
) -> dict[tuple[int, int, int], tuple[list[set[str]], list[str]]]:
    """Solve several XYZ compositions in one DP pass.

    Radius searches share almost all intermediate states. Solving the complete
    requested frontier once makes end-to-end walk-forward telemetry practical
    while producing the same deterministic ticket as an isolated solve.
    """
    if len(games) != 14 or not distributions or any(
            not is_xyz_distribution_valid(*point) for point in distributions):
        raise ValueError("XYZ exige 14 jogos e distribuições viáveis")
    actions = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))
    limits = tuple(max(point[rank] for point in distributions) for rank in range(3))
    minimums = tuple(min(point[rank] for point in distributions) for rank in range(3))
    # A base-6 action code preserves deterministic lexicographic ties without
    # repeatedly copying 14-element path tuples in every intermediate state.
    # Z is derivable at layer i: Z = (i + doubles) - X - Y. Keeping only
    # (X, Y, doubles) substantially reduces tuple allocation in the hot loop.
    deltas = ((1, 0, 0), (0, 1, 0), (0, 0, 0),
              (1, 1, 1), (1, 0, 1), (0, 1, 1))
    states: dict[tuple[int, int, int], tuple[float, int]] = {(0, 0, 0): (0.0, 0)}
    for game_index, game in enumerate(games):
        remaining_games = 13 - game_index
        forced_result = team_result(game, "FLAMENGO")
        forced_rank = game.ranking.index(forced_result) if forced_result else None
        updated = {}
        action_coverages = tuple(sum(game.probabilities[game.ranking[rank]]
                                     for rank in action) for action in actions)
        for (used_x, used_y, used_doubles), (score, path_code) in states.items():
            for action_index, action in enumerate(actions):
                if forced_rank is not None and forced_rank not in action:
                    continue
                dx, dy, dd = deltas[action_index]
                key = used_x + dx, used_y + dy, used_doubles + dd
                next_z = game_index + 1 + key[2] - key[0] - key[1]
                rank_counts = key[0], key[1], next_z
                if any(rank_counts[rank] > limits[rank] for rank in range(3)) or key[2] > 5:
                    continue
                if key[2] + remaining_games < 5 or any(
                        rank_counts[rank] + remaining_games < minimums[rank]
                        for rank in range(3)):
                    continue
                candidate = (score + action_coverages[action_index],
                             path_code * 6 + action_index)
                if key not in updated or candidate[0] > updated[key][0] or (
                        candidate[0] == updated[key][0] and candidate[1] < updated[key][1]):
                    updated[key] = candidate
        states = updated
    solved = {}
    for distribution in distributions:
        final = (distribution[0], distribution[1], 5)
        if final not in states:
            raise ValueError("Hard Constraint do Flamengo torna a distribuição XYZ inviável")
        path_code = states[final][1]
        action_indexes = [0] * 14
        for index in range(13, -1, -1):
            path_code, action_indexes[index] = divmod(path_code, 6)
        ticket = [set(game.ranking[rank] for rank in actions[action_index])
                  for game, action_index in zip(games, action_indexes)]
        if sorted(map(len, ticket)) != [1] * 9 + [2] * 5 or sum(map(len, ticket)) != 19:
            raise AssertionError("otimizador XYZ violou a estrutura 9/5/0")
        notes = [f"XYZ solicitado/efetivo: {xyz_distribution_id(distribution)}"]
        for game, pick in zip(games, ticket):
            flamengo_win = team_result(game, "FLAMENGO")
            if flamengo_win:
                if flamengo_win not in pick:
                    raise AssertionError("otimizador XYZ não cobriu a vitória do Flamengo")
                notes.append(f"FLAMENGO jogo {game.jogo}: vitória {flamengo_win} coberta")
        solved[distribution] = ticket, notes
    return solved


def _best_distribution_assignment(games: list[Match], top2_count: int,
                                  value) -> tuple[set[int], set[int]]:
    """Solve the rank-placement problem with a small deterministic DP."""
    if not 0 <= top2_count <= 5:
        raise ValueError("a distribuição segura exige entre 0 e 5 marcas Top2")
    forced: dict[int, int] = {}
    for index, game in enumerate(games):
        flamengo_win = team_result(game, "FLAMENGO")
        if flamengo_win and flamengo_win != game.ranking[0]:
            forced[index] = game.ranking.index(flamengo_win) + 1
    # At an extreme distribution the Flamengo hard constraint can make the
    # nominal composition mathematically infeasible. Use the nearest safe
    # composition rather than ever removing Top1 or Flamengo's victory.
    if 2 in forced.values() and top2_count == 0:
        top2_count = 1
    if 3 in forced.values() and top2_count == 5:
        top2_count = 4
    top3_count = 5 - top2_count
    # State values contain score and the sequence of rank choices (0, 2, 3).
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {(0, 0): (0.0, ())}
    for index, game in enumerate(games):
        updated = {}
        for (used_top2, used_top3), (score, choices) in states.items():
            choices_for_game = (forced[index],) if index in forced else (0, 2, 3)
            for rank in choices_for_game:
                next_top2 = used_top2 + (rank == 2)
                next_top3 = used_top3 + (rank == 3)
                if next_top2 > top2_count or next_top3 > top3_count:
                    continue
                candidate = (score + (value(game, rank) if rank else 0.0),
                             choices + (rank,))
                key = (next_top2, next_top3)
                # Lexicographically smaller choices keep earlier games on ties.
                if key not in updated or candidate[0] > updated[key][0] or (
                        candidate[0] == updated[key][0] and candidate[1] < updated[key][1]):
                    updated[key] = candidate
        states = updated
    choices = states[(top2_count, top3_count)][1]
    doubles = {index for index, rank in enumerate(choices) if rank}
    top2 = {index for index, rank in enumerate(choices) if rank == 2}
    return doubles, top2


def distribution_ticket(games: list[Match], top2_count: int
                        ) -> tuple[list[set[str]], list[str]]:
    """Optimize the placement of one of the six safe rank distributions.

    The score is the total covered pre-match probability.  Actual results are
    deliberately inaccessible here, making this function safe for prediction
    and backtesting.  All candidates pass through the constraint engine.
    """
    if len(games) != 14:
        raise ValueError("a distribuição exige exatamente 14 jogos")
    double_indexes, top2_indexes = _best_distribution_assignment(
        games, top2_count,
        lambda game, rank: game.probabilities[game.ranking[rank - 1]],
    )
    marks = {index: games[index].ranking[1 if index in top2_indexes else 2]
             for index in double_indexes}
    # A safe distribution must preserve Top1; the Palmeiras preference is soft
    # and therefore disabled here when it would replace that mandatory mark.
    ticket, notes = build_ticket(games, double_indexes, palmeiras_threshold=-1.0,
                                 second_marks=marks)
    if any(game.ranking[0] not in pick for game, pick in zip(games, ticket)):
        raise AssertionError("distribuição segura removeu Top1")
    return ticket, notes


def oracle_distribution_ticket(games: list[Match], top2_count: int
                               ) -> list[set[str]]:
    """Return the best retrospective placement for one safe distribution."""
    _ticket_hits(games, [{game.ranking[0]} for game in games])
    double_indexes, top2_indexes = _best_distribution_assignment(
        games, top2_count,
        lambda game, rank: float(game.actual == game.ranking[rank - 1]),
    )
    marks = {index: games[index].ranking[1 if index in top2_indexes else 2]
             for index in double_indexes}
    ticket = build_ticket(games, double_indexes, palmeiras_threshold=-1.0,
                          second_marks=marks)[0]
    if any(game.ranking[0] not in pick for game, pick in zip(games, ticket)):
        raise AssertionError("oracle de distribuição removeu Top1")
    return ticket


def distribution_backtest(contests: dict[int, list[Match]],
                          minimum_history: int = MIN_WALK_FORWARD_CONTESTS
                          ) -> dict[str, object]:
    """Evaluate every safe distribution and its retrospective upper bound."""
    contest_ids = sorted(contests)
    if not 1 <= minimum_history < len(contest_ids):
        raise ValueError("janela inicial inválida para backtest de distribuição")
    hits = {f"14/{top2}/{top3}": [] for _, top2, top3 in SAFE_DISTRIBUTIONS}
    oracle_hits = {key: [] for key in hits}
    selected_oracle_hits: list[int] = []
    selected_distribution_hits: list[int] = []
    for position in range(minimum_history, len(contest_ids)):
        assert contest_ids[position - 1] < contest_ids[position]
        games = contests[contest_ids[position]]
        contest_oracles = []
        for _, top2, top3 in SAFE_DISTRIBUTIONS:
            key = f"14/{top2}/{top3}"
            ticket = distribution_ticket(games, top2)[0]
            oracle_ticket = oracle_distribution_ticket(games, top2)
            hits[key].append(_ticket_hits(games, ticket))
            oracle_hits[key].append(_ticket_hits(games, oracle_ticket))
            contest_oracles.append((oracle_hits[key][-1], top2, key))
        oracle_hit, _, oracle_key = max(contest_oracles)
        selected_oracle_hits.append(oracle_hit)
        selected_distribution_hits.append(hits[oracle_key][-1])
    return {
        "test_contests": len(contest_ids) - minimum_history,
        "no_future_information": True,
        "distributions": {key: {**_hit_summary(values),
                                  "median": float(statistics.median(values)),
                                  "stddev": round(statistics.pstdev(values), 6)}
                          for key, values in hits.items()},
        "oracle_by_distribution": {key: _hit_summary(values)
                                    for key, values in oracle_hits.items()},
        "oracle_distribution": _hit_summary(selected_oracle_hits),
        "distribution_regret": _regret_summary([
            oracle - selected for oracle, selected in
            zip(selected_oracle_hits, selected_distribution_hits)
        ]),
        "diagnostic_only": True,
    }


def _distribution_selection_key(summary: dict[str, int | float],
                                distance: int = 0) -> tuple[float, ...]:
    """Apply the project's ticket-level objective with conservative ties."""
    return (
        float(summary["p13_plus_empirical"]), float(summary["14"]),
        float(summary["13"]), float(summary["p12_plus_empirical"]),
        float(summary["12"]), float(summary["mean"]),
        -float(summary["stddev"]), -distance,
    )


def xyz_distribution_backtest(
    contests: dict[int, list[Match]],
    center: tuple[int, int, int] = (9, 5, 5),
    radius: int = 1,
    minimum_history: int = MIN_WALK_FORWARD_CONTESTS,
) -> dict[str, object]:
    """Evaluate the controlled XYZ space without using outcomes to build tickets.

    The oracle chooses only among already-frozen, probability-optimized XYZ
    tickets for each contest. It is therefore explicitly retrospective and is
    kept separate from the operational selection.
    """
    contest_ids = sorted(contests)
    if not 1 <= minimum_history < len(contest_ids):
        raise ValueError("janela inicial inválida para backtest XYZ")
    distributions = generate_xyz_radius(center, radius)
    hit_history = {xyz_distribution_id(point): [] for point in distributions}
    oracle_hits: list[int] = []
    oracle_usage = {key: 0 for key in hit_history}

    for position in range(minimum_history, len(contest_ids)):
        assert contest_ids[position - 1] < contest_ids[position]
        games = contests[contest_ids[position]]
        contest_hits: list[tuple[int, int, str]] = []
        tickets = _xyz_distribution_tickets(games, distributions)
        for point in distributions:
            key = xyz_distribution_id(point)
            ticket, _ = tickets[point]
            hits = _ticket_hits(games, ticket)
            hit_history[key].append(hits)
            distance = sum(abs(a - b) for a, b in zip(center, point)) // 2
            contest_hits.append((hits, -distance, key))
        best_hits, _, best_key = max(contest_hits)
        oracle_hits.append(best_hits)
        oracle_usage[best_key] += 1

    summaries = {
        key: {
            **_hit_summary(values),
            "median": float(statistics.median(values)),
            "stddev": round(statistics.pstdev(values), 6),
        }
        for key, values in hit_history.items()
    }
    oracle = _hit_summary(oracle_hits)
    regrets = {
        key: _regret_summary([best - fixed for best, fixed in zip(oracle_hits, values)])
        for key, values in hit_history.items()
    }
    points_by_id = {xyz_distribution_id(point): point for point in distributions}
    best_xyz = max(summaries, key=lambda key: _distribution_selection_key(
        summaries[key], sum(abs(a - b) for a, b in zip(center, points_by_id[key])) // 2
    ))

    # SAFE is evaluated over exactly the same contests, allowing meaningful
    # ticket-level deltas rather than comparisons between different samples.
    safe = distribution_backtest(contests, minimum_history)
    best_safe = max(safe["distributions"], key=lambda key:
                    _distribution_selection_key(safe["distributions"][key]))
    xyz_summary, safe_summary = summaries[best_xyz], safe["distributions"][best_safe]
    return {
        "center": xyz_distribution_id(center), "radius": radius,
        "test_contests": len(oracle_hits), "no_future_information": True,
        "distributions": summaries,
        "regret_by_distribution": regrets,
        "oracle_xyz": oracle, "oracle_xyz_usage": oracle_usage,
        "xyz_vs_safe": {
            "best_safe": best_safe, "best_xyz": best_xyz,
            "delta_p13_plus": round(float(xyz_summary["p13_plus_empirical"]) -
                                    float(safe_summary["p13_plus_empirical"]), 8),
            "delta_p12_plus": round(float(xyz_summary["p12_plus_empirical"]) -
                                    float(safe_summary["p12_plus_empirical"]), 8),
            "delta_mean": round(float(xyz_summary["mean"]) -
                                float(safe_summary["mean"]), 6),
        },
        "diagnostic_only": True,
    }


def oracle_tickets(games: list[Match], baseline: list[set[str]]) -> dict[str, list[set[str]]]:
    """Return retrospective allocator, selector and full-oracle tickets.

    These tickets are diagnostics only: real outcomes are read explicitly and
    are never exposed through ``ticket_for_policy`` or the prediction pipeline.
    Every candidate still goes through ``build_ticket``, preserving 9/5/0 and
    the Flamengo hard constraint.
    """
    if len(games) != 14 or sorted(map(len, baseline)) != [1] * 9 + [2] * 5:
        raise ValueError("oráculos exigem um ticket baseline 9/5/0")
    _ticket_hits(games, baseline)
    baseline_indexes = {i for i, pick in enumerate(baseline) if len(pick) == 2}

    def best_marks(indexes: set[int]) -> dict[int, str]:
        marks: dict[int, str] = {}
        for index in indexes:
            game = games[index]
            candidates = (game.ranking[1], game.ranking[2])
            # Evaluate through the constraint engine because Flamengo coverage
            # can replace a requested mark. Ties retain the safer Top2 baseline.
            marks[index] = max(candidates, key=lambda mark: (
                int(game.actual in constrained_pick(game, True, 0.03, mark)[0]),
                -candidates.index(mark),
            ))
        return marks

    def best_indexes(second_marks: dict[int, str] | None = None) -> set[int]:
        gains = []
        for index, game in enumerate(games):
            single = constrained_pick(game, False, 0.03)[0]
            mark = second_marks[index] if second_marks else None
            double = constrained_pick(game, True, 0.03, mark)[0]
            gains.append(int(game.actual in double) - int(game.actual in single))
        return set(sorted(range(14), key=lambda index: (-gains[index], index))[:5])

    allocator = build_ticket(games, best_indexes())[0]
    selector = build_ticket(games, baseline_indexes,
                            second_marks=best_marks(baseline_indexes))[0]
    all_best_marks = best_marks(set(range(14)))
    full_indexes = best_indexes(all_best_marks)
    full = build_ticket(games, full_indexes,
                        second_marks={i: all_best_marks[i] for i in full_indexes})[0]
    return {"allocator": allocator, "selector": selector, "full": full}


def _hit_summary(hits: list[int]) -> dict[str, int | float]:
    return {
        "14": sum(value == 14 for value in hits),
        "13": sum(value == 13 for value in hits),
        "12": sum(value == 12 for value in hits),
        "p13_plus_empirical": round(sum(value >= 13 for value in hits) / len(hits), 8),
        "p12_plus_empirical": round(sum(value >= 12 for value in hits) / len(hits), 8),
        "mean": round(statistics.fmean(hits), 6),
    }


def _regret_summary(regrets: list[int]) -> dict[str, int | float]:
    return {
        "mean_regret": round(statistics.fmean(regrets), 6),
        "median_regret": float(statistics.median(regrets)),
        "regret_0_rate": round(sum(value == 0 for value in regrets) / len(regrets), 8),
        "regret_1_rate": round(sum(value == 1 for value in regrets) / len(regrets), 8),
        "regret_2plus_rate": round(sum(value >= 2 for value in regrets) / len(regrets), 8),
        "max_regret": max(regrets),
    }


def oracle_decomposition(contests: dict[int, list[Match]], policy: str,
                         minimum_history: int = MIN_WALK_FORWARD_CONTESTS
                         ) -> dict[str, object]:
    """Measure structural headroom on the same out-of-sample contests."""
    contest_ids = sorted(contests)
    ordered = [contests[key] for key in contest_ids]
    if not 1 <= minimum_history < len(ordered):
        raise ValueError("janela inicial inválida para decomposição oracle")
    hits = {name: [] for name in ("baseline", "allocator", "selector", "full")}
    for index in range(minimum_history, len(ordered)):
        assert contest_ids[index - 1] < contest_ids[index]
        rates = position_rank_hit_rates(ordered[:index])
        baseline, _ = ticket_for_policy(ordered[index], policy, rates)
        oracle = oracle_tickets(ordered[index], baseline)
        hits["baseline"].append(_ticket_hits(ordered[index], baseline))
        for name, ticket in oracle.items():
            hits[name].append(_ticket_hits(ordered[index], ticket))
    return {
        "test_contests": len(hits["baseline"]),
        "diagnostic_only": True,
        "baseline_policy": policy,
        **{name: _hit_summary(values) for name, values in hits.items()},
        "regret": {
            name: _regret_summary([oracle - baseline for oracle, baseline in
                                   zip(hits[name], hits["baseline"])])
            for name in ("allocator", "selector", "full")
        },
    }


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
    nested_recovery = nested_walk_forward_second_mark(contests)
    model = {
        "version": 14, "selected_policy": selected,
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
        "allocator_diagnostics": allocator_diagnostics(contests),
        "oracle_decomposition": oracle_decomposition(contests, selected),
        "distribution_backtest": distribution_backtest(contests),
        "xyz_distribution_backtest": xyz_distribution_backtest(contests),
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
            "nested_walk_forward": nested_recovery,
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
