"""Historical selection of the five-double allocation policy."""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

from scripts.common import Match, group_contests, normalize_team, read_matches, ticket_metrics

PROBABILITY_POLICIES = ("gain", "uncertainty", "margin", "ratio")
HISTORICAL_POLICIES = ("hist_top1", "hist_top2")
POLICIES = (*PROBABILITY_POLICIES, *HISTORICAL_POLICIES)
MIN_WALK_FORWARD_CONTESTS = 30


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
                     palmeiras_threshold: float) -> tuple[set[str], list[str]]:
    """Create one constrained pick without changing its number of markings."""
    selection = set(game.ranking[:2] if is_double else game.ranking[:1])
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
                 palmeiras_threshold: float = 0.03) -> tuple[list[set[str]], list[str]]:
    """Build every ticket through the same constraints-aware pipeline."""
    if len(games) != 14 or len(double_indexes) != 5 or not double_indexes <= set(range(14)):
        raise ValueError("o ticket exige 14 jogos e exatamente 5 índices de duplos")
    ticket: list[set[str]] = []
    notes: list[str] = []
    for i, game in enumerate(games):
        selection, pick_notes = constrained_pick(
            game, i in double_indexes, palmeiras_threshold
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
                          minimum_history: int = MIN_WALK_FORWARD_CONTESTS
                          ) -> dict[str, dict[str, int]]:
    """Compare every policy prospectively, never learning from the test contest."""
    ordered = [contests[key] for key in sorted(contests)]
    if not 1 <= minimum_history < len(ordered):
        raise ValueError("janela inicial inválida para walk-forward")
    strategies = (*POLICIES, "exact")
    evaluations = {policy: {"14": 0, "13": 0, "hits": 0}
                   for policy in strategies}
    for index in range(minimum_history, len(ordered)):
        rates = position_rank_hit_rates(ordered[:index])
        games = ordered[index]
        for policy in strategies:
            ticket, _ = ticket_for_policy(games, policy, rates)
            hits = sum(game.actual in selection
                       for game, selection in zip(games, ticket))
            evaluations[policy]["hits"] += hits
            evaluations[policy]["14"] += hits == 14
            evaluations[policy]["13"] += hits == 13
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


def train(history_path: str, model_path: str) -> dict[str, object]:
    contests = group_contests(read_matches(history_path, require_actual=True))
    evaluations = walk_forward_backtest(contests)
    # Main goal first, then perfect tickets and aggregate hits as stable tie-breakers.
    strategies = (*POLICIES, "exact")
    selected = max(strategies, key=lambda p: (
        evaluations[p]["13"] + evaluations[p]["14"],
        evaluations[p]["14"], evaluations[p]["hits"], -strategies.index(p),
    ))
    rank_hits = [0, 0, 0]
    for games in contests.values():
        for game in games:
            rank_hits[game.ranking.index(game.actual)] += 1
    model = {
        "version": 4, "selected_policy": selected,
        "contests_evaluated": len(contests), "policy_backtest": evaluations,
        "walk_forward": {
            "minimum_history": MIN_WALK_FORWARD_CONTESTS,
            "test_contests": len(contests) - MIN_WALK_FORWARD_CONTESTS,
            "no_future_information": True,
        },
        "position_rank_hit_rates": position_rank_hit_rates(list(contests.values())),
        "rank_hit_rates": [round(count / sum(rank_hits), 6) for count in rank_hits],
        "probability_diagnostics": probability_diagnostics(contests),
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return model
