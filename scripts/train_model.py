"""Historical selection of the five-double allocation policy."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.common import Match, group_contests, read_matches

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


def build_ticket(games: list[Match], policy: str) -> list[set[str]]:
    double_indexes = set(sorted(range(14), key=lambda i: (-priority(games[i], policy), i))[:5])
    return [set(game.ranking[:2] if i in double_indexes else game.ranking[:1])
            for i, game in enumerate(games)]


def train(history_path: str, model_path: str) -> dict[str, object]:
    contests = group_contests(read_matches(history_path, require_actual=True))
    evaluations = {}
    for policy in POLICIES:
        totals = {"14": 0, "13": 0, "hits": 0}
        for games in contests.values():
            ticket = build_ticket(games, policy)
            hits = sum(game.actual in selection for game, selection in zip(games, ticket))
            totals["hits"] += hits
            totals["14"] += hits == 14
            totals["13"] += hits == 13
        evaluations[policy] = totals
    # Main goal first, then perfect tickets and aggregate hits as stable tie-breakers.
    selected = max(POLICIES, key=lambda p: (
        evaluations[p]["13"] + evaluations[p]["14"],
        evaluations[p]["14"], evaluations[p]["hits"], -POLICIES.index(p),
    ))
    rank_hits = [0, 0, 0]
    for games in contests.values():
        for game in games:
            rank_hits[game.ranking.index(game.actual)] += 1
    model = {
        "version": 1, "selected_policy": selected,
        "contests_evaluated": len(contests), "policy_backtest": evaluations,
        "rank_hit_rates": [round(count / sum(rank_hits), 6) for count in rank_hits],
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return model
