"""Validate historical data and expose rank-relative one-hot observations."""

from __future__ import annotations

from scripts.common import group_contests, read_matches


def preprocess(path: str) -> list[dict[str, object]]:
    matches = read_matches(path, require_actual=True)
    group_contests(matches)
    records = []
    for match in matches:
        rank = match.ranking.index(match.actual) + 1
        records.append({
            "concurso": match.concurso, "jogo": match.jogo,
            "top1_hit": int(rank == 1), "top2_hit": int(rank == 2),
            "top3_hit": int(rank == 3),
        })
    return records
