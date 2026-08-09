"""Ticket optimizer, constraints, audit telemetry and CSV export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.common import Match, read_matches
from scripts.train_model import exact_ticket, heuristic_ticket, ticket_metrics_for


def optimize(games: list[Match], policy: str) -> tuple[list[set[str]], list[str]]:
    if len(games) != 14:
        raise ValueError("o próximo concurso deve conter exatamente 14 jogos")
    return exact_ticket(games) if policy == "exact" else heuristic_ticket(games, policy)


def predict(input_path: str, model_path: str, output_path: str, verbose: bool = True) -> list[dict[str, object]]:
    games = sorted(read_matches(input_path), key=lambda game: game.jogo)
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    ticket, notes = optimize(games, model["selected_policy"])
    gains = [game.probabilities[game.ranking[1]] for game in games]
    gain_ranks = {index: rank for rank, index in enumerate(
        sorted(range(14), key=lambda i: (-gains[i], i)), 1
    )}
    rows = []
    for index, (game, selection) in enumerate(zip(games, ticket)):
        ranking = game.ranking
        ordered_pick = "".join(result for result in ("1", "X", "2") if result in selection)
        rows.append({
            "concurso": game.concurso, "jogo": game.jogo, "mandante": game.mandante,
            "visitante": game.visitante, "p_1": f"{game.probabilities['1']:.6f}",
            "p_x": f"{game.probabilities['X']:.6f}", "p_2": f"{game.probabilities['2']:.6f}",
            **{f"top{i}_result": result for i, result in enumerate(ranking, 1)},
            **{f"top{i}_prob": f"{game.probabilities[result]:.6f}" for i, result in enumerate(ranking, 1)},
            "tipo_aposta": "DUPLO" if len(selection) == 2 else "SECO", "palpite": ordered_pick,
            "covered_probability": f"{sum(game.probabilities[r] for r in selection):.6f}",
            "double_gain": f"{gains[index]:.6f}" if len(selection) == 2 else "",
            "double_rank": gain_ranks[index] if len(selection) == 2 else "",
        })
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys(), delimiter=";", lineterminator="\n"
        )
        writer.writeheader(); writer.writerows(rows)
    if verbose:
        print(f"[INFO] Concurso: {games[0].concurso}")
        print("[INFO] Estratégia: 9 secos / 5 duplos / 0 triplos")
        print(f"[OPT] Estratégia histórica selecionada: {model['selected_policy']}")
        for row in rows:
            print(f"[JOGO {row['jogo']:02d}] {row['mandante']} x {row['visitante']}")
            print(f"  p(1): {row['p_1']}  p(X): {row['p_x']}  p(2): {row['p_2']}")
            print(f"  Top1/Top2/Top3: {row['top1_result']}/{row['top2_result']}/{row['top3_result']}")
            print(f"  Escolha: {row['tipo_aposta']} {row['palpite']}")
            if row["tipo_aposta"] == "DUPLO":
                print(f"  Ganho marginal: +{float(row['double_gain']):.4%} "
                      f"(ranking de ganho {row['double_rank']}/14)")
        for note in notes: print(f"[CONSTRAINT] {note}")
        metrics = ticket_metrics_for(games, ticket)
        print(f"[METRIC] P(14): {metrics['p14']:.6%}")
        print(f"[METRIC] P(13): {metrics['p13']:.6%}")
        print(f"[METRIC] P(>=13): {metrics['p13_plus']:.6%}")
        print(f"[METRIC] E[acertos]: {metrics['expected_hits']:.4f}")
        print("[FINAL] 9 secos / 5 duplos / 0 triplos — 19 marcações")
    return rows
