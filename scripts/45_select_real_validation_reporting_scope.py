from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_REAL_DIR = PROJECT_ROOT / "results_real"
TABLES_DIR = RESULTS_REAL_DIR / "tables"

PRIMARY_DATASET = "waqd2024"
PRIMARY_TARGET = "PM25_p95_rel_p90_h1"
PRIMARY_SCENARIO = "minimum_support"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def make_main_table(summary: pd.DataFrame) -> pd.DataFrame:
    main = summary[
        (summary["scenario"] == PRIMARY_SCENARIO)
        & (summary["dataset"] == PRIMARY_DATASET)
        & (summary["target"] == PRIMARY_TARGET)
    ].copy()
    main["configuration"] = main.apply(configuration_label, axis=1)
    columns = [
        "model",
        "configuration",
        "countries",
        "rows",
        "events",
        "pooled_event_rate",
        "mean_pr_auc",
        "mean_f1",
        "mean_recall",
        "mean_precision",
        "mean_false_alarm_rate",
    ]
    model_order = {"logistic_regression": 1, "random_forest": 2, "xgboost": 3}
    config_order = {"Base": 1, "Local features": 2, "Country-class balanced": 3}
    main["model_order"] = main["model"].map(model_order)
    main["config_order"] = main["configuration"].map(config_order)
    return main.sort_values(["model_order", "config_order"])[columns]


def direction_for_target(summary: pd.DataFrame, dataset: str, target: str) -> dict[str, object]:
    data = summary[
        (summary["scenario"] == PRIMARY_SCENARIO)
        & (summary["dataset"] == dataset)
        & (summary["target"] == target)
        & (summary["model"] == "xgboost")
    ].copy()
    if not data.empty:
        data["configuration"] = data.apply(configuration_label, axis=1)
    if data.empty:
        return {
            "dataset": dataset,
            "target": target,
            "countries": 0,
            "rows": 0,
            "events": 0,
            "local_pr_auc_improves": "not_available",
            "weighting_recall_increases": "not_available",
            "weighting_false_alarm_increases": "not_available",
            "compatible_findings": 0,
            "reporting_role": "not_available",
            "note": "No filtered XGBoost LOCO result found.",
        }

    indexed = data.set_index("configuration")
    required = {"Base", "Local features", "Country-class balanced"}
    if not required.issubset(set(indexed.index)):
        return {
            "dataset": dataset,
            "target": target,
            "countries": int(data["countries"].max()),
            "rows": int(data["rows"].max()),
            "events": int(data["events"].max()),
            "local_pr_auc_improves": "not_available",
            "weighting_recall_increases": "not_available",
            "weighting_false_alarm_increases": "not_available",
            "compatible_findings": 0,
            "reporting_role": "not_available",
            "note": "Incomplete XGBoost configuration set.",
        }

    base = indexed.loc["Base"]
    local = indexed.loc["Local features"]
    weighted = indexed.loc["Country-class balanced"]

    local_pr_auc_improves = bool(local["mean_pr_auc"] > base["mean_pr_auc"])
    weighting_recall_increases = bool(weighted["mean_recall"] > local["mean_recall"])
    weighting_false_alarm_increases = bool(
        weighted["mean_false_alarm_rate"] > local["mean_false_alarm_rate"]
    )
    compatible_findings = sum(
        [
            local_pr_auc_improves,
            weighting_recall_increases,
            weighting_false_alarm_increases,
        ]
    )

    if dataset == PRIMARY_DATASET and target == PRIMARY_TARGET:
        reporting_role = "main_external_validation"
        note = "Use as the main real-data validation result."
    elif dataset == "openaq" and target == PRIMARY_TARGET:
        reporting_role = "sensitivity_check"
        note = "Use as an independent real-data sensitivity check, not as the main narrative."
    elif "PM10" in target:
        reporting_role = "supplementary_pm10"
        note = "Use only as a short supplementary robustness note if space allows."
    else:
        reporting_role = "sensitivity_check"
        note = "Use only as supporting evidence."

    return {
        "dataset": dataset,
        "target": target,
        "countries": int(local["countries"]),
        "rows": int(local["rows"]),
        "events": int(local["events"]),
        "base_pr_auc": float(base["mean_pr_auc"]),
        "local_pr_auc": float(local["mean_pr_auc"]),
        "weighted_pr_auc": float(weighted["mean_pr_auc"]),
        "local_pr_auc_improves": "yes" if local_pr_auc_improves else "no",
        "local_pr_auc_delta": float(local["mean_pr_auc"] - base["mean_pr_auc"]),
        "local_recall": float(local["mean_recall"]),
        "weighted_recall": float(weighted["mean_recall"]),
        "weighting_recall_increases": "yes" if weighting_recall_increases else "no",
        "weighting_recall_delta": float(weighted["mean_recall"] - local["mean_recall"]),
        "local_false_alarm_rate": float(local["mean_false_alarm_rate"]),
        "weighted_false_alarm_rate": float(weighted["mean_false_alarm_rate"]),
        "weighting_false_alarm_increases": "yes"
        if weighting_false_alarm_increases
        else "no",
        "weighting_false_alarm_delta": float(
            weighted["mean_false_alarm_rate"] - local["mean_false_alarm_rate"]
        ),
        "compatible_findings": compatible_findings,
        "reporting_role": reporting_role,
        "note": note,
    }


def make_direction_table(summary: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        summary[["dataset", "target"]]
        .drop_duplicates()
        .sort_values(["dataset", "target"])
        .itertuples(index=False, name=None)
    )
    rows = [direction_for_target(summary, dataset, target) for dataset, target in pairs]
    return pd.DataFrame(rows)


def configuration_label(row: pd.Series) -> str:
    if row["feature_regime"] == "base" and row["weighting_strategy"] == "none":
        return "Base"
    if row["feature_regime"] == "local" and row["weighting_strategy"] == "none":
        return "Local features"
    if (
        row["feature_regime"] == "local"
        and row["weighting_strategy"] == "country_class_balanced"
    ):
        return "Country-class balanced"
    return f"{row['feature_regime']} + {row['weighting_strategy']}"


def make_scope_table(direction: pd.DataFrame) -> pd.DataFrame:
    main = direction[
        (direction["dataset"] == PRIMARY_DATASET)
        & (direction["target"] == PRIMARY_TARGET)
    ]
    openaq = direction[
        (direction["dataset"] == "openaq")
        & (direction["target"] == PRIMARY_TARGET)
    ]
    pm10 = direction[direction["target"].str.contains("PM10", na=False)]

    def evidence(data: pd.DataFrame) -> str:
        if data.empty:
            return "not available"
        matched = int((data["compatible_findings"] >= 2).sum())
        total = len(data)
        mean_checks = data["compatible_findings"].mean()
        return f"{matched}/{total} settings match at least 2/3 checks; mean checks={mean_checks:.2f}"

    rows = [
        {
            "component": "Main real-data validation",
            "dataset": PRIMARY_DATASET,
            "target": PRIMARY_TARGET,
            "include_in_body": "yes",
            "recommended_text_role": (
                "Primary external validation using WAQD2024, PM2.5, h=1, LOCO, "
                "and the minimum-support fold filter."
            ),
            "directional_evidence": evidence(main),
        },
        {
            "component": "OpenAQ sensitivity",
            "dataset": "openaq",
            "target": PRIMARY_TARGET,
            "include_in_body": "briefly",
            "recommended_text_role": (
                "Mention as an independent sensitivity check; keep numeric detail outside "
                "the main narrative unless needed."
            ),
            "directional_evidence": evidence(openaq),
        },
        {
            "component": "PM10 analyses",
            "dataset": "waqd2024/openaq",
            "target": "PM10_p95_rel_p90_h1 and PM10_p95_rel_p90_h3",
            "include_in_body": "optional",
            "recommended_text_role": (
                "Use as supplementary robustness evidence only; avoid giving PM10 the same "
                "weight as PM2.5 in the paper body."
            ),
            "directional_evidence": evidence(pm10),
        },
    ]

    return pd.DataFrame(rows)


def main_markdown(main: pd.DataFrame) -> str:
    header = (
        "| Model | Configuration | Countries | Rows | Events | PR-AUC | F1 | "
        "Recall | Precision | FAR |\n"
    )
    separator = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    rows = []
    for _, row in main.iterrows():
        rows.append(
            "| "
            f"{row['model']} | {row['configuration']} | "
            f"{int(row['countries'])} | {int(row['rows'])} | {int(row['events'])} | "
            f"{format_metric(row['mean_pr_auc'])} | {format_metric(row['mean_f1'])} | "
            f"{format_metric(row['mean_recall'])} | {format_metric(row['mean_precision'])} | "
            f"{format_metric(row['mean_false_alarm_rate'])} |"
        )
    return header + separator + "\n".join(rows) + "\n"


def direction_markdown(direction: pd.DataFrame) -> str:
    columns = [
        "dataset",
        "target",
        "countries",
        "local_pr_auc_improves",
        "weighting_recall_increases",
        "weighting_false_alarm_increases",
        "compatible_findings",
        "reporting_role",
    ]
    header = (
        "| Dataset | Target | Countries | Local PR-AUC improves | Weighting recall "
        "increases | Weighting FAR increases | Compatible findings | Role |\n"
    )
    separator = "|---|---|---:|---:|---:|---:|---:|---|\n"
    rows = []
    for _, row in direction[columns].iterrows():
        rows.append(
            "| "
            f"{row['dataset']} | {row['target']} | {int(row['countries'])} | "
            f"{row['local_pr_auc_improves']} | {row['weighting_recall_increases']} | "
            f"{row['weighting_false_alarm_increases']} | "
            f"{int(row['compatible_findings'])} | {row['reporting_role']} |"
        )
    return header + separator + "\n".join(rows) + "\n"


def article_note(main: pd.DataFrame, direction: pd.DataFrame) -> str:
    xgb = main[main["model"] == "xgboost"].set_index("configuration")
    rf = main[main["model"] == "random_forest"].set_index("configuration")
    openaq_primary = direction[
        (direction["dataset"] == "openaq") & (direction["target"] == PRIMARY_TARGET)
    ]
    pm10 = direction[direction["target"].str.contains("PM10", na=False)]

    openaq_note = "not available"
    if not openaq_primary.empty:
        row = openaq_primary.iloc[0]
        openaq_note = (
            f"{row['compatible_findings']}/3 directional checks matched in OpenAQ "
            f"for the same PM2.5 h=1 target."
        )

    pm10_note = (
        f"{int((pm10['compatible_findings'] >= 2).sum())}/{len(pm10)} PM10 sensitivity "
        "settings matched at least two of the three directional checks."
        if not pm10.empty
        else "PM10 sensitivity was not available."
    )

    return (
        "Recommended real-data reporting scope\n"
        "=====================================\n\n"
        "Use WAQD2024 PM2.5 h=1 under LOCO with the minimum-support fold filter as the "
        "main external validation result. This setting keeps the article focused on one "
        "real-data question: whether the synthetic benchmark conclusions are directionally "
        "visible in independent monitoring data.\n\n"
        "Main result to report:\n"
        f"- XGBoost local features increase PR-AUC from "
        f"{format_metric(xgb.loc['Base', 'mean_pr_auc'])} to "
        f"{format_metric(xgb.loc['Local features', 'mean_pr_auc'])}.\n"
        f"- XGBoost country-class weighting increases recall from "
        f"{format_metric(xgb.loc['Local features', 'mean_recall'])} to "
        f"{format_metric(xgb.loc['Country-class balanced', 'mean_recall'])}, with FAR "
        f"rising from {format_metric(xgb.loc['Local features', 'mean_false_alarm_rate'])} "
        f"to {format_metric(xgb.loc['Country-class balanced', 'mean_false_alarm_rate'])}.\n"
        f"- Random Forest local features provide the strongest PR-AUC among the default "
        f"threshold models in this table: "
        f"{format_metric(rf.loc['Local features', 'mean_pr_auc'])}.\n\n"
        "Sensitivity handling:\n"
        f"- OpenAQ should be mentioned as a sensitivity check, not as a second main result: "
        f"{openaq_note}\n"
        f"- PM10 should remain supplementary: {pm10_note}\n\n"
        "Recommended wording:\n"
        "In the real-data validation, WAQD2024 PM2.5 h=1 was treated as the primary "
        "external setting, while OpenAQ and PM10 variants were used as sensitivity checks. "
        "This avoids overextending the limited and irregular real-data coverage while "
        "testing whether the main directional patterns persist outside the synthetic "
        "benchmark.\n"
    )


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    full_summary = read_csv(TABLES_DIR / "real_loco_all_vs_min_support_summary.csv")

    main_table = make_main_table(full_summary)
    direction_table = make_direction_table(full_summary)
    scope_table = make_scope_table(direction_table)

    main_csv = TABLES_DIR / "real_external_validation_main_result.csv"
    main_md = TABLES_DIR / "real_external_validation_main_result.md"
    direction_csv = TABLES_DIR / "real_external_validation_sensitivity_direction.csv"
    direction_md = TABLES_DIR / "real_external_validation_sensitivity_direction.md"
    scope_csv = TABLES_DIR / "real_external_validation_reporting_scope.csv"
    note_path = TABLES_DIR / "real_external_validation_reporting_note.txt"

    main_table.to_csv(main_csv, index=False)
    main_md.write_text(main_markdown(main_table), encoding="utf-8")
    direction_table.to_csv(direction_csv, index=False)
    direction_md.write_text(direction_markdown(direction_table), encoding="utf-8")
    scope_table.to_csv(scope_csv, index=False)
    note_path.write_text(article_note(main_table, direction_table), encoding="utf-8")

    print(f"Saved main external validation result: {main_csv}")
    print(f"Saved main external validation markdown: {main_md}")
    print(f"Saved sensitivity direction table: {direction_csv}")
    print(f"Saved reporting scope table: {scope_csv}")
    print(f"Saved reporting note: {note_path}")
    print(main_markdown(main_table))


if __name__ == "__main__":
    main()
