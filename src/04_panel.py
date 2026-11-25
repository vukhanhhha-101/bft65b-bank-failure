"""PART 2b - clean, point-in-time bank-quarter panel.

Each report dated r is available at pred_date = end of quarter r+1.
Label: y=1 when first FAILURE date is in (pred_date, pred_date + 1 year].
Rows whose bank has already failed by pred_date are excluded from the risk set.
"""
import json
import polars as pl
from config import *

def main():
    A = pl.col
    if PUB_LAG_MONTHS <= 0 or PUB_LAG_MONTHS % 3 != 0:
        raise ValueError("PUB_LAG_MONTHS must be a positive multiple of 3.")

    # 1. One bank-quarter row, ordered before all temporal operations.
    fin = (
        scan_panel()
        .unique(subset=["CERT", "REPDTE"], keep="last", maintain_order=True)
        .sort(["CERT", "REPDTE"])
        .with_columns(QIDX)
        .with_columns(
            (A("qidx") - A("qidx").shift(1).over("CERT") == 1).alias("contig")
        )
        .collect()
    )
    n_raw = fin.height

    # 2. De-cumulate year-to-date flow variables.
    # Q1 is the reported YTD value; Q2-Q4 require the immediately prior quarter.
    fin = fin.with_columns([
        pl.when(A("REPDTE").dt.month() == 3)
          .then(A(c))
          .when(A("contig"))
          .then(A(c) - A(c).shift(1).over("CERT"))
          .otherwise(None)
          .alias(f"{c}_Q")
        for c in FLOW
    ])

    # 3. Minimal hard validity rule; other extremes are handled in train-fitted preprocessing.
    fin = fin.filter(A("ASSET").is_not_null() & (A("ASSET") > 0))
    n_after_asset = fin.height
    n_drop_asset = n_raw - n_after_asset

    # 4. First valid-CERT FAILURE event only. Assistance is not a failure label.
    fd = failure_dates().select(["CERT", "FAILDATE"])
    last = fin.group_by("CERT").agg(A("REPDTE").max().alias("LAST_REPDTE"))
    max_report_date = fin["REPDTE"].max()

    fin = (
        fin.join(fd, on="CERT", how="left")
        .join(last, on="CERT", how="left")
        .with_columns(
            pl.when(A("FAILDATE").is_not_null())
              .then(pl.lit("failure"))
              .when(A("LAST_REPDTE") < pl.lit(max_report_date))
              .then(pl.lit("not_reporting_in_latest_quarter"))
              .otherwise(pl.lit("reporting_in_latest_quarter"))
              .alias("EXIT_TYPE")
        )
    )

    # 5. Remove report rows dated on/after failure; these cannot be predictors.
    n_before_event_filter = fin.height
    fin = fin.filter(A("FAILDATE").is_null() | (A("REPDTE") < A("FAILDATE")))
    n_drop_on_after_failure = n_before_event_filter - fin.height

    # 6. Prediction date and prediction-quarter index.
    fin = fin.with_columns(
        A("REPDTE").dt.offset_by(f"{PUB_LAG_MONTHS}mo").dt.month_end().alias("pred_date"),
        (A("qidx") + PUB_LAG_MONTHS // 3).alias("pq")
    )
    lag_days = (fin["pred_date"] - fin["REPDTE"]).dt.total_days()
    assert lag_days.min() >= MIN_PUB_LAG_DAYS, (
        f"Minimum publication lag is {lag_days.min()} days; "
        f"required at least {MIN_PUB_LAG_DAYS}."
    )

    # Diagnostics comparing the old report-date label with the point-in-time label.
    old_label = (
        (A("FAILDATE") > A("REPDTE"))
        & (A("FAILDATE") <= A("REPDTE").dt.offset_by("1y"))
    ).fill_null(False)
    within_reporting_horizon = A("REPDTE") <= pl.lit(LAST_REPORT_DATE)
    n_old_positive_rows = fin.filter(old_label & within_reporting_horizon).height
    already_failed_at_decision = (A("FAILDATE") <= A("pred_date")).fill_null(False)
    n_old_positive_removed_by_lag = fin.filter(
        old_label & already_failed_at_decision & within_reporting_horizon
    ).height

    # Exclude banks already failed by the decision date.
    n_before_risk_filter = fin.height
    fin = fin.filter(~already_failed_at_decision)

    # 7. Outcome window is strictly after prediction date and up to 1 year after.
    fin = fin.with_columns(
        (
            (A("FAILDATE") > A("pred_date"))
            & (A("FAILDATE") <= A("pred_date").dt.offset_by("1y"))
        ).fill_null(False).cast(pl.Int8).alias("y")
    ).drop("contig").sort(["CERT", "REPDTE"])

    # Keep complete historical panel for downstream point-in-time feature construction.
    fin.write_parquet(PANEL_CLEAN, compression="zstd")

    # Modelling rows stop at the final report that maps to LAST_LABEL_DATE.
    mod = fin.filter(
        (A("REPDTE") <= pl.lit(LAST_REPORT_DATE))
        & (A("pred_date") <= pl.lit(LAST_LABEL_DATE))
    )
    dev = mod.filter(A("pq") <= DEV_END_Q)
    test = mod.filter(A("pq").is_between(TEST_START_Q, TEST_END_Q))

    def positive_banks(d):
        return d.filter(A("y") == 1)["CERT"].n_unique()

    exit_counts = (
        fin.group_by("CERT")
        .agg(A("EXIT_TYPE").first().alias("EXIT_TYPE"))
        .group_by("EXIT_TYPE").len()
        .sort("EXIT_TYPE")
        .to_dicts()
    )

    summary = {
        "rows_raw_panel": n_raw,
        "rows_dropped_asset_le0_or_null": n_drop_asset,
        "rows_dropped_report_on_or_after_failure": n_drop_on_after_failure,
        "rows_dropped_already_failed_at_prediction_date": n_before_risk_filter - fin.height,
        "rows_clean_panel": fin.height,
        "rows_modelling_table": mod.height,
        "banks_clean_panel": fin["CERT"].n_unique(),
        "exit_types_bank_level": exit_counts,
        "dev_rows": dev.height,
        "dev_failed_banks": positive_banks(dev),
        "test_rows": test.height,
        "test_failed_banks": positive_banks(test),
        "positive_rate_dev": dev["y"].mean(),
        "positive_rate_test": test["y"].mean(),
        "old_report_date_positive_rows_before_cutoff": n_old_positive_rows,
        "old_positive_rows_removed_by_publication_lag": n_old_positive_removed_by_lag,
        "pub_lag_months": PUB_LAG_MONTHS,
        "minimum_observed_pub_lag_days": int(lag_days.min()),
        "failure_only": FAILURE_ONLY,
        "last_report_date": str(LAST_REPORT_DATE),
        "last_prediction_date": str(LAST_LABEL_DATE)
    }
    (REPORTS / "panel_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    assert summary["test_failed_banks"] >= 50, (
        "Test period must contain at least 50 failed banks; "
        "check date/index definitions before proceeding."
    )

if __name__ == "__main__":
    main()
