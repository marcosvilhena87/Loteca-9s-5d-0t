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
    allocator = model["allocator_diagnostics"]
    for left, right in (("uncertainty", "gain"), ("uncertainty", "top2_probability"),
                        ("uncertainty", "ratio"), ("uncertainty", "exact")):
        key = next(key for key in allocator["overlap_mean_of_5"]
                   if set(key.split("__")) == {left, right})
        print(f"[ALLOCATOR OVERLAP] {left} x {right}: "
              f"{allocator['overlap_mean_of_5'][key]:.3f} / 5")
    comparison = allocator["pairwise"]["gain__uncertainty"]
    print(f"[PAIRWISE] gain vs uncertainty: {comparison['wins']} vitórias | "
          f"{comparison['ties']} empates | {comparison['losses']} derrotas | "
          f"delta médio {comparison['mean_delta_hits']:+.4f}")
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
    nested = model["error_recovery"]["nested_walk_forward"]
    print(f"[NESTED RECOVERY] {nested['test_contests']} concursos | "
          f"delta P13+ {nested['delta_p13_plus']:+.2%} | "
          f"delta P12+ {nested['delta_p12_plus']:+.2%} | "
          f"thresholds {nested['threshold_usage']}")
    oracle = model["oracle_decomposition"]
    print("[ORACLE DECOMPOSITION]")
    for name in ("baseline", "allocator", "selector", "full"):
        result = oracle[name]
        print(f"  {name}: P13+ {result['p13_plus_empirical']:.2%} | "
              f"P12+ {result['p12_plus_empirical']:.2%} | média {result['mean']:.4f}")
    for name, regret in oracle["regret"].items():
        print(f"[REGRET {name.upper()}] média {regret['mean_regret']:.4f} | "
              f"zero {regret['regret_0_rate']:.2%} | 2+ {regret['regret_2plus_rate']:.2%} | "
              f"máximo {regret['max_regret']}")
    distributions = model["distribution_backtest"]
    print("[DISTRIBUTION BACKTEST]")
    for name, result in distributions["distributions"].items():
        print(f"  {name}: P13+ {result['p13_plus_empirical']:.2%} | "
              f"P12+ {result['p12_plus_empirical']:.2%} | média {result['mean']:.4f}")
    distribution_oracle = distributions["oracle_distribution"]
    print(f"[ORACLE DISTRIBUTION] P13+ "
          f"{distribution_oracle['p13_plus_empirical']:.2%} | "
          f"P12+ {distribution_oracle['p12_plus_empirical']:.2%}")
    xyz = model["xyz_distribution_backtest"]
    print("[XYZ DISTRIBUTION BACKTEST]")
    for name, result in xyz["distributions"].items():
        print(f"  {name}: 14 {result['14']} | 13 {result['13']} | 12 {result['12']} | "
              f"P13+ {result['p13_plus_empirical']:.2%} | "
              f"P12+ {result['p12_plus_empirical']:.2%} | média {result['mean']:.4f} | "
              f"mediana {result['median']:.1f} | desvio {result['stddev']:.4f}")
    print("[XYZ OBJECTIVE COMPARISON]")
    for name, result in xyz["objective_comparison"].items():
        coverage = result["coverage"]
        direct = result["direct_p13"]
        print(f"  {name}: coverage P13+ {coverage['p13_plus_empirical']:.2%} | "
              f"direct P13+ {direct['p13_plus_empirical']:.2%} | "
              f"delta {result['delta_p13_plus_empirical']:+.2%} | "
              f"modelo {result['coverage_model_p13_plus']:.6%} -> "
              f"{result['direct_model_p13_plus']:.6%} | "
              f"ganho {result['mean_modeled_p13_plus_gain']:+.6%}")
    comparison = xyz["xyz_vs_safe"]
    print(f"[XYZ VS SAFE] best_safe {comparison['best_safe']} | "
          f"best_xyz {comparison['best_xyz']} | "
          f"delta P13+ {comparison['delta_p13_plus']:+.2%} | "
          f"delta P12+ {comparison['delta_p12_plus']:+.2%} | "
          f"delta média {comparison['delta_mean']:+.4f}")
    xyz_oracle = xyz["retrospective_frozen_selection"]
    print(f"[XYZ RETROSPECTIVE FROZEN SELECTION] P13+ "
          f"{xyz_oracle['p13_plus_empirical']:.2%} | "
          f"P12+ {xyz_oracle['p12_plus_empirical']:.2%} | "
          f"média {xyz_oracle['mean']:.4f}")
    print("[XYZ RETROSPECTIVE FROZEN SELECTION USAGE] "
          f"{xyz['retrospective_frozen_selection_usage']}")
    for name, regret in xyz["regret_by_distribution"].items():
        print(f"[REGRET {name}] média {regret['mean_regret']:.4f} | "
              f"mediana {regret['median_regret']:.1f} | zero {regret['regret_0_rate']:.2%} | "
              f"um {regret['regret_1_rate']:.2%} | 2+ {regret['regret_2plus_rate']:.2%} | "
              f"máximo {regret['max_regret']}")
    true_xyz = model["true_oracle_xyz"]
    profile = model["actual_rank_profile"]
    print("[ACTUAL RANK PROFILE] "
          f"médias T1/T2/T3 {profile['mean_top1']:.3f}/"
          f"{profile['mean_top2']:.3f}/{profile['mean_top3']:.3f} | "
          f"medianas {profile['median_top1']:.0f}/"
          f"{profile['median_top2']:.0f}/{profile['median_top3']:.0f}")
    print(f"[ACTUAL RANK PROFILE MOST COMMON] {profile['most_common_profiles']}")
    print("[TRUE ORACLE XYZ BY DISTRIBUTION]")
    for name, result in true_xyz["by_distribution"].items():
        print(f"  {name}: P13+ {result['p13_plus_empirical']:.2%} | "
              f"P12+ {result['p12_plus_empirical']:.2%} | média {result['mean']:.4f}")
    print("[XYZ ORACLE FEASIBILITY]")
    for name, result in true_xyz["feasibility"].items():
        print(f"  {name}: feasible14 {result['feasible_14_rate']:.2%} | "
              f"feasible13+ {result['feasible_13_plus_rate']:.2%}")
    result = true_xyz["overall"]
    print(f"[TRUE ORACLE XYZ] P13+ {result['p13_plus_empirical']:.2%} | "
          f"P12+ {result['p12_plus_empirical']:.2%} | média {result['mean']:.4f} | "
          f"usage {true_xyz['usage']}")
    comparison = true_xyz["comparison"]
    print("[ORACLE CEILING COMPARISON] "
          f"Distribution {comparison['oracle_distribution']['p13_plus_empirical']:.2%} | "
          f"TrueXYZ {result['p13_plus_empirical']:.2%} | "
          f"Full {comparison['oracle_full']['p13_plus_empirical']:.2%}")
    predict(args.next_contest, args.model, args.output)


if __name__ == "__main__":
    main()
