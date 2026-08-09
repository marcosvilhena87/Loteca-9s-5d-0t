"""Historical selection of the five-double allocation policy."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from scripts.common import Match, group_contests, normalize_team, read_matches, ticket_metrics

POLICIES = ("gain", "uncertainty", "margin", "ratio")


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
    evaluations = {}
    for policy in (*POLICIES, "exact"):
        totals = {"14": 0, "13": 0, "hits": 0}
        for games in contests.values():
            ticket, _ = exact_ticket(games) if policy == "exact" else heuristic_ticket(games, policy)
            hits = sum(game.actual in selection for game, selection in zip(games, ticket))
            totals["hits"] += hits
            totals["14"] += hits == 14
            totals["13"] += hits == 13
        evaluations[policy] = totals
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
        "version": 2, "selected_policy": selected,
        "contests_evaluated": len(contests), "policy_backtest": evaluations,
        "rank_hit_rates": [round(count / sum(rank_hits), 6) for count in rank_hits],
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return model
