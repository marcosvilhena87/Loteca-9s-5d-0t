"""Ticket optimizer, constraints, audit telemetry and CSV export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.common import Match, normalize_team, read_matches, threshold_probability
from scripts.train_model import build_ticket


def team_result(match: Match, needle: str) -> str | None:
    home, away = normalize_team(match.mandante), normalize_team(match.visitante)
    return "1" if needle in home else ("2" if needle in away else None)


def optimize(games: list[Match], policy: str) -> tuple[list[set[str]], list[str]]:
    if len(games) != 14:
        raise ValueError("o próximo concurso deve conter exatamente 14 jogos")
    ticket = build_ticket(games, policy)
    notes: list[str] = []
    for i, game in enumerate(games):
        flamengo_win = team_result(game, "FLAMENGO")
        if flamengo_win and flamengo_win not in ticket[i]:
            removed = min(ticket[i], key=lambda result: game.probabilities[result])
            ticket[i].remove(removed)
            ticket[i].add(flamengo_win)
        if flamengo_win:
            notes.append(f"FLAMENGO jogo {game.jogo}: vitória {flamengo_win} coberta")

        palmeiras_win = team_result(game, "PALMEIRAS")
        if palmeiras_win in ticket[i]:
            alternatives = [result for result in game.ranking if result != palmeiras_win]
            replacement = alternatives[0]
            loss = game.probabilities[palmeiras_win] - game.probabilities[replacement]
            if loss <= 0.03 and len(ticket[i]) == 1:
                ticket[i] = {replacement}
                notes.append(f"PALMEIRAS jogo {game.jogo}: vitória excluída (perda {loss:.3f})")
            else:
                notes.append(f"PALMEIRAS jogo {game.jogo}: preferência não aplicada (perda {loss:.3f})")
    return ticket, notes


def predict(input_path: str, model_path: str, output_path: str, verbose: bool = True) -> list[dict[str, object]]:
    games = sorted(read_matches(input_path), key=lambda game: game.jogo)
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    ticket, notes = optimize(games, model["selected_policy"])
    rows = []
    for game, selection in zip(games, ticket):
        ranking = game.ranking
        ordered_pick = "".join(result for result in ("1", "X", "2") if result in selection)
        rows.append({
            "concurso": game.concurso, "jogo": game.jogo, "mandante": game.mandante,
            "visitante": game.visitante, "p_1": f"{game.probabilities['1']:.6f}",
            "p_x": f"{game.probabilities['X']:.6f}", "p_2": f"{game.probabilities['2']:.6f}",
            **{f"top{i}_result": result for i, result in enumerate(ranking, 1)},
            **{f"top{i}_prob": f"{game.probabilities[result]:.6f}" for i, result in enumerate(ranking, 1)},
            "tipo_aposta": "DUPLO" if len(selection) == 2 else "SECO", "palpite": ordered_pick,
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
        print(f"[OPT] Política histórica selecionada: {model['selected_policy']}")
        for row in rows:
            print(f"[JOGO {row['jogo']:02d}] {row['mandante']} x {row['visitante']}")
            print(f"  p(1): {row['p_1']}  p(X): {row['p_x']}  p(2): {row['p_2']}")
            print(f"  Top1/Top2/Top3: {row['top1_result']}/{row['top2_result']}/{row['top3_result']}")
            print(f"  Escolha: {row['tipo_aposta']} {row['palpite']}")
        for note in notes: print(f"[CONSTRAINT] {note}")
        covered = [sum(game.probabilities[r] for r in pick) for game, pick in zip(games, ticket)]
        print(f"[METRIC] Probabilidade estimada de >=13: {threshold_probability(covered):.6%}")
        print("[FINAL] 9 secos / 5 duplos / 0 triplos — 19 marcações")
    return rows
