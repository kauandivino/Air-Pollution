from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_REAL_DIR = PROJECT_ROOT / "results_real"
TABLES_DIR = RESULTS_DIR / "tables"
REAL_TABLES_DIR = RESULTS_REAL_DIR / "tables"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def format_float(value: float) -> str:
    return f"{value:.3f}"


def get_synthetic_protocol_finding() -> tuple[str, str]:
    table = read_csv(TABLES_DIR / "protocol_comparison_model_summary.csv")
    model = "random_forest"
    rf = table[table["model"] == model].set_index("split_name")

    random_pr = float(rf.loc["random_split", "mean_pr_auc"])
    temporal_pr = float(rf.loc["temporal_split", "mean_pr_auc"])
    loco_pr = float(rf.loc["leave_one_country_out", "mean_pr_auc"])

    finding = yes_no(random_pr > loco_pr and temporal_pr > loco_pr)
    evidence = (
        f"RF PR-AUC: random={format_float(random_pr)}, "
        f"temporal={format_float(temporal_pr)}, LOCO={format_float(loco_pr)}"
    )
    return finding, evidence


def get_real_protocol_finding(real_primary: pd.DataFrame) -> tuple[str, str]:
    protocol = read_csv(REAL_TABLES_DIR / "real_primary_protocol_comparison.csv")
    rf_protocol = protocol[
        (protocol["model"] == "random_forest")
        & (protocol["configuration"] == "Base")
    ].set_index("protocol")
    rf_filtered = real_primary[
        (real_primary["scenario"] == "minimum_support")
        & (real_primary["model"] == "random_forest")
        & (real_primary["configuration"] == "Base")
    ].iloc[0]

    random_pr = float(rf_protocol.loc["random_split", "mean_pr_auc"])
    temporal_pr = float(rf_protocol.loc["temporal_split", "mean_pr_auc"])
    loco_pr = float(rf_filtered["mean_pr_auc"])

    if random_pr > loco_pr and temporal_pr > loco_pr:
        finding = "yes"
    elif random_pr > loco_pr or temporal_pr > loco_pr:
        finding = "partial"
    else:
        finding = "no"

    evidence = (
        f"RF PR-AUC: random={format_float(random_pr)}, "
        f"temporal={format_float(temporal_pr)}, filtered LOCO={format_float(loco_pr)}"
    )
    return finding, evidence


def get_synthetic_local_features_finding() -> tuple[str, str]:
    table = read_csv(TABLES_DIR / "local_features_base_vs_local_summary.csv")
    row = table[
        (table["target"] == "extreme_abs_151_h3")
        & (table["model"] == "xgboost")
        & (table["metric"] == "pr_auc")
    ].iloc[0]

    base = float(row["base_mean"])
    local = float(row["local_mean"])
    delta = float(row["mean_delta"])
    finding = yes_no(local > base)
    evidence = (
        f"XGBoost h3 PR-AUC: base={format_float(base)}, "
        f"local={format_float(local)}, delta={format_float(delta)}"
    )
    return finding, evidence


def get_real_local_features_finding(real_primary: pd.DataFrame) -> tuple[str, str]:
    data = real_primary[
        (real_primary["scenario"] == "minimum_support")
        & (real_primary["model"] == "xgboost")
    ].set_index("configuration")

    base = float(data.loc["Base", "mean_pr_auc"])
    local = float(data.loc["Local features", "mean_pr_auc"])
    delta = local - base
    finding = yes_no(local > base)
    evidence = (
        f"XGBoost PR-AUC: base={format_float(base)}, "
        f"local={format_float(local)}, delta={format_float(delta)}"
    )
    return finding, evidence


def get_synthetic_weighting_finding() -> tuple[str, str]:
    table = read_csv(TABLES_DIR / "domain_weighting_vs_local_summary.csv")
    row = table[
        (table["target"] == "extreme_abs_151_h3")
        & (table["model"] == "xgboost")
        & (table["weighting_strategy"] == "country_class_balanced")
        & (table["metric"] == "recall")
    ].iloc[0]

    local = float(row["local_mean"])
    weighted = float(row["weighted_mean"])
    delta = float(row["mean_delta"])
    finding = yes_no(weighted > local)
    evidence = (
        f"XGBoost h3 recall: local={format_float(local)}, "
        f"weighted={format_float(weighted)}, delta={format_float(delta)}"
    )
    return finding, evidence


def get_real_weighting_finding(real_primary: pd.DataFrame) -> tuple[str, str]:
    data = real_primary[
        (real_primary["scenario"] == "minimum_support")
        & (real_primary["model"] == "xgboost")
    ].set_index("configuration")

    local = float(data.loc["Local features", "mean_recall"])
    weighted = float(data.loc["Country-class balanced", "mean_recall"])
    delta = weighted - local
    finding = yes_no(weighted > local)
    evidence = (
        f"XGBoost recall: local={format_float(local)}, "
        f"weighted={format_float(weighted)}, delta={format_float(delta)}"
    )
    return finding, evidence


def get_synthetic_false_alarm_finding() -> tuple[str, str]:
    table = read_csv(TABLES_DIR / "domain_weighting_vs_local_summary.csv")
    row = table[
        (table["target"] == "extreme_abs_151_h3")
        & (table["model"] == "xgboost")
        & (table["weighting_strategy"] == "country_class_balanced")
        & (table["metric"] == "false_alarm_rate")
    ].iloc[0]

    local = float(row["local_mean"])
    weighted = float(row["weighted_mean"])
    delta = float(row["mean_delta"])
    finding = yes_no(weighted > local)
    evidence = (
        f"XGBoost h3 FAR: local={format_float(local)}, "
        f"weighted={format_float(weighted)}, delta={format_float(delta)}"
    )
    return finding, evidence


def get_real_false_alarm_finding(real_primary: pd.DataFrame) -> tuple[str, str]:
    data = real_primary[
        (real_primary["scenario"] == "minimum_support")
        & (real_primary["model"] == "xgboost")
    ].set_index("configuration")

    local = float(data.loc["Local features", "mean_false_alarm_rate"])
    weighted = float(data.loc["Country-class balanced", "mean_false_alarm_rate"])
    delta = weighted - local
    finding = yes_no(weighted > local)
    evidence = (
        f"XGBoost FAR: local={format_float(local)}, "
        f"weighted={format_float(weighted)}, delta={format_float(delta)}"
    )
    return finding, evidence


def make_markdown_table(comparison: pd.DataFrame) -> str:
    columns = ["finding", "synthetic", "real", "interpretation"]
    header = "| Finding | Synthetic | Real | Interpretation |\n"
    separator = "|---|---:|---:|---|\n"
    rows = []
    for _, row in comparison[columns].iterrows():
        rows.append(
            f"| {row['finding']} | {row['synthetic']} | {row['real']} | {row['interpretation']} |"
        )
    return header + separator + "\n".join(rows) + "\n"


def main() -> None:
    real_primary = read_csv(REAL_TABLES_DIR / "real_loco_primary_all_vs_min_support.csv")

    rows: list[dict[str, str]] = []

    synthetic, synthetic_evidence = get_synthetic_protocol_finding()
    real, real_evidence = get_real_protocol_finding(real_primary)
    rows.append(
        {
            "finding": "Random/temporal split exceeds LOCO",
            "synthetic": synthetic,
            "real": real,
            "synthetic_evidence": synthetic_evidence,
            "real_evidence": real_evidence,
            "interpretation": (
                "Protocol optimism is strong in the synthetic benchmark; in the real "
                "filtered analysis it is clear for random split but weaker for temporal split."
            ),
        }
    )

    synthetic, synthetic_evidence = get_synthetic_local_features_finding()
    real, real_evidence = get_real_local_features_finding(real_primary)
    rows.append(
        {
            "finding": "Local features improve PR-AUC",
            "synthetic": synthetic,
            "real": real,
            "synthetic_evidence": synthetic_evidence,
            "real_evidence": real_evidence,
            "interpretation": (
                "Local normalization improves ranking in both settings under the selected "
                "XGBoost comparison."
            ),
        }
    )

    synthetic, synthetic_evidence = get_synthetic_weighting_finding()
    real, real_evidence = get_real_weighting_finding(real_primary)
    rows.append(
        {
            "finding": "Country-class weighting increases recall",
            "synthetic": synthetic,
            "real": real,
            "synthetic_evidence": synthetic_evidence,
            "real_evidence": real_evidence,
            "interpretation": (
                "The recall-oriented behavior of country-class weighting is reproduced "
                "in the filtered real-data validation."
            ),
        }
    )

    synthetic, synthetic_evidence = get_synthetic_false_alarm_finding()
    real, real_evidence = get_real_false_alarm_finding(real_primary)
    rows.append(
        {
            "finding": "Country-class weighting increases false alarms",
            "synthetic": synthetic,
            "real": real,
            "synthetic_evidence": synthetic_evidence,
            "real_evidence": real_evidence,
            "interpretation": (
                "The higher-recall configuration also increases the false-alarm rate, "
                "preserving the operational trade-off."
            ),
        }
    )

    comparison = pd.DataFrame(rows)
    csv_path = REAL_TABLES_DIR / "synthetic_vs_real_finding_bridge.csv"
    md_path = REAL_TABLES_DIR / "synthetic_vs_real_finding_bridge.md"

    comparison.to_csv(csv_path, index=False)
    md_path.write_text(make_markdown_table(comparison), encoding="utf-8")

    print(f"Saved synthetic vs real comparison: {csv_path}")
    print(f"Saved markdown table: {md_path}")
    print(make_markdown_table(comparison))


if __name__ == "__main__":
    main()
