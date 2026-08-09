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
    args = parser.parse_args()
    model = train(args.history, args.model)
    print(f"[TRAIN] {model['contests_evaluated']} concursos; Top hits: {model['rank_hit_rates']}")
    print(f"[BACKTEST] {model['policy_backtest']}")
    diagnostics = model["probability_diagnostics"]
    print(f"[CALIBRATION] Brier: {diagnostics['multiclass_brier']:.6f} | "
          f"Log Loss: {diagnostics['log_loss']:.6f} | ECE: {diagnostics['ece']:.6f}")
    predict(args.next_contest, args.model, args.output)


if __name__ == "__main__":
    main()
