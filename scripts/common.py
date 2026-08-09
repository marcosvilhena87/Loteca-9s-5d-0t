"""Shared domain and CSV helpers for the Loteca pipeline."""

from __future__ import annotations

import csv
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path

RESULTS = ("1", "X", "2")
TIE_PRIORITY = {"1": 2, "2": 1, "X": 0}


def parse_decimal(value: str) -> float:
    """Parse the comma-decimal format explicitly required by the data contract."""
    return float(value.strip().replace(".", "").replace(",", "."))


def normalize_team(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return "".join(ch for ch in value.upper() if ch.isalnum())


@dataclass
class Match:
    concurso: int
    jogo: int
    mandante: str
    visitante: str
    probabilities: dict[str, float]
    actual: str | None = None

    @property
    def ranking(self) -> list[str]:
        return sorted(
            RESULTS,
            key=lambda result: (self.probabilities[result], TIE_PRIORITY[result]),
            reverse=True,
        )


def read_matches(path: str | Path, require_actual: bool = False) -> list[Match]:
    """Read semicolon CSV, accepting UTF-8 and the legacy CP1252 dataset."""
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    rows = csv.DictReader(text.splitlines(), delimiter=";")
    matches: list[Match] = []
    for row in rows:
        probabilities = {
            "1": parse_decimal(row["p(1)"]),
            "X": parse_decimal(row["p(x)"]),
            "2": parse_decimal(row["p(2)"]),
        }
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError(f"probabilidades inválidas no jogo {row['Jogo']}")
        probabilities = {key: value / total for key, value in probabilities.items()}
        actuals = [result for result in RESULTS if row.get(result, "0").strip() == "1"]
        if require_actual and len(actuals) != 1:
            raise ValueError(f"resultado real inválido no concurso {row['Concurso']}")
        matches.append(Match(
            concurso=int(row["Concurso"]), jogo=int(row["Jogo"]),
            mandante=row["Mandante"], visitante=row["Visitante"],
            probabilities=probabilities, actual=actuals[0] if len(actuals) == 1 else None,
        ))
    return matches


def group_contests(matches: list[Match]) -> dict[int, list[Match]]:
    grouped: dict[int, list[Match]] = {}
    for match in matches:
        grouped.setdefault(match.concurso, []).append(match)
    for games in grouped.values():
        games.sort(key=lambda game: game.jogo)
        if len(games) != 14:
            raise ValueError("cada concurso deve conter exatamente 14 jogos")
    return grouped


def threshold_probability(probabilities: list[float], target: int = 13) -> float:
    """Poisson-binomial P(hits >= target), assuming independent matches."""
    distribution = [1.0] + [0.0] * len(probabilities)
    for probability in probabilities:
        for hits in range(len(probabilities), 0, -1):
            distribution[hits] = (
                distribution[hits] * (1 - probability)
                + distribution[hits - 1] * probability
            )
        distribution[0] *= 1 - probability
    return sum(distribution[target:])


def ticket_metrics(probabilities: list[float]) -> dict[str, float]:
    """Return the exact Poisson-binomial ticket metrics.

    Match outcomes are assumed to be independent, as documented in the README.
    """
    if len(probabilities) != 14 or any(not 0 <= value <= 1 for value in probabilities):
        raise ValueError("métricas exigem 14 probabilidades entre zero e um")
    p14 = math.prod(probabilities)
    # Prefix/suffix products keep P(13) linear and also handle zero coverage safely.
    prefix = [1.0]
    for probability in probabilities:
        prefix.append(prefix[-1] * probability)
    suffix = [1.0] * 15
    for i in range(13, -1, -1):
        suffix[i] = suffix[i + 1] * probabilities[i]
    p13 = sum((1.0 - probability) * prefix[i] * suffix[i + 1]
              for i, probability in enumerate(probabilities))
    return {"p14": p14, "p13": p13, "p13_plus": p13 + p14,
            "expected_hits": sum(probabilities)}
