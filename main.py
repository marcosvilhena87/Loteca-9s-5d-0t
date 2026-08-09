"""Run training, historical strategy selection and final prediction."""

import argparse

from scripts.predict_results import predict
from scripts.train_model import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Otimizador Loteca ML")
    parser.add_argument("--history", default="data/concursos_anteriores.csv")
    parser.add_argument("--next", dest="next_contest", default="data/proximo_concurso.csv")
    parser.add_argument("--model", default="models/model.json")
    parser.add_argument("--output", default="output/predictions.csv")
    parser.add_argument("--backtest-output", default="output/backtest.csv")
    args = parser.parse_args()
    model = train(args.history, args.model, args.backtest_output)
    print(f"[TRAIN] {model['contests_evaluated']} concursos; Top hits: {model['rank_hit_rates']}")
    print(f"[BACKTEST] {model['policy_backtest']}")
    print(f"[WALK-FORWARD] {model['walk_forward']['test_contests']} concursos de teste; "
          f"janela inicial: {model['walk_forward']['minimum_history']}")
    diagnostics = model["probability_diagnostics"]
    print(f"[CALIBRATION] Brier: {diagnostics['multiclass_brier']:.6f} | "
          f"Log Loss: {diagnostics['log_loss']:.6f} | ECE: {diagnostics['ece']:.6f}")
    for metric, audit in model["top1_reliability"]["walk_forward_disagreement"].items():
        print(f"[DISAGREEMENT] {metric}: {audit['cases']} casos | "
              f"baseline {audit['baseline_wins']} x histórico {audit['historical_wins']} | "
              f"neutros {audit['neutral']} | win rate {audit['historical_win_rate']:.2%}")
    meta = model["top1_meta"]
    audit = meta["disagreement"]
    print(f"[TOP1-META] Brier baseline {meta['baseline_brier']:.6f} x "
          f"meta {meta['meta_brier']:.6f}")
    print(f"[DISAGREEMENT] p_top1_meta: {audit['cases']} casos | "
          f"baseline {audit['baseline_wins']} x meta {audit['meta_wins']} | "
          f"neutros {audit['neutral']} | win rate {audit['meta_win_rate']:.2%}")
    recovery = model["error_recovery"]["walk_forward_disagreement"]
    print(f"[SECOND-MARK DISAGREEMENT] {recovery['cases']} casos | "
          f"Top2 {recovery['top2_baseline_wins']} x recovery {recovery['recovery_wins']} | "
          f"win rate {recovery['recovery_win_rate']:.2%} | "
          f"seletor final: {model['selected_second_mark']}")
    for threshold, result in recovery["threshold_results"].items():
        low, high = result["recovery_win_rate_ci95"]
        print(f"[THRESHOLD RECOVERY {threshold}] {result['cases']} trocas | "
              f"Top2 {result['top2_baseline_wins']} x recovery {result['recovery_wins']} | "
              f"win rate {result['recovery_win_rate']:.2%} | IC95% [{low:.2%}, {high:.2%}]")
    predict(args.next_contest, args.model, args.output)


if __name__ == "__main__":
    main()
