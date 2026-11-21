"""PART 2b - clean, point-in-time bank-quarter panel with exit type and label.
y(i,q) = 1{FAILDATE in (q, q + 1 year]}; mergers / other exits are 0."""
import json
import polars as pl
from config import *


def main():
    A = pl.col
    fin = (scan_panel().unique(subset=["CERT", "REPDTE"], keep="last", maintain_order=True)   # 1. de-duplicate
           .sort(["CERT", "REPDTE"]).with_columns(QIDX)
           .with_columns((A("qidx") - A("qidx").shift(1).over("CERT") == 1).alias("contig")))
    # 2. de-cumulate year-to-date flows (Q1 = as reported; Q2..Q4 = difference with previous quarter, only if contiguous)
    fin = fin.with_columns([
        pl.when(A("REPDTE").dt.month() == 3).then(A(c))
          .when(A("contig")).then(A(c) - A(c).shift(1).over("CERT")).otherwise(None).alias(f"{c}_Q") for c in FLOW])
    fin = fin.collect()
    n0 = fin.height
    # 3. minimal validity filter (everything else is handled by train-fitted winsorising in the model pipeline)
    fin = fin.filter(A("ASSET") > 0)
    n_invalid = n0 - fin.height

    fd = failure_dates()
    last = fin.group_by("CERT").agg(A("REPDTE").max().alias("LAST_REPDTE"))
    maxd = fin["REPDTE"].max()
    fin = (fin.join(fd.select(["CERT", "FAILDATE"]), on="CERT", how="left").join(last, on="CERT", how="left")
              .with_columns(pl.when(A("FAILDATE").is_not_null()).then(pl.lit("failure"))
                              .when(A("LAST_REPDTE") < maxd).then(pl.lit("other_exit"))
                              .otherwise(pl.lit("active")).alias("EXIT_TYPE")))   # diagnostic only - NEVER a feature (leakage)
    # 4. drop reports dated on/after the failure date (cannot exist point-in-time)
    n1 = fin.height
    fin = fin.filter(A("FAILDATE").is_null() | (A("REPDTE") < A("FAILDATE")))
    n_after = n1 - fin.height
    # 5. label
    fin = fin.with_columns(
        ((A("FAILDATE") > A("REPDTE")) & (A("FAILDATE") <= A("REPDTE").dt.offset_by("1y"))).fill_null(False).cast(pl.Int8).alias("y"),
        # diagnostic: failure happened before the Call Report could have been published (~45 days)
        (A("FAILDATE") <= A("REPDTE") + pl.duration(days=45)).fill_null(False).alias("fail_before_avail"))
    fin = fin.drop(["contig"]).sort(["CERT", "REPDTE"])
    fin.write_parquet(PANEL_CLEAN, compression="zstd")

    mod = fin.filter(A("REPDTE") <= LAST_LABEL_DATE)
    dev = mod.filter(A("qidx") <= DEV_END_Q); test = mod.filter(A("qidx").is_between(TEST_START_Q, TEST_END_Q))
    nf = lambda d: d.filter(A("y") == 1)["CERT"].n_unique()
    summary = dict(rows_raw_panel=n0, rows_dropped_asset_le0=n_invalid, rows_dropped_report_after_failure=n_after,
                   rows_clean_panel=fin.height, rows_modelling_table=mod.height, banks=fin["CERT"].n_unique(),
                   exit_types=fin.group_by("CERT").agg(A("EXIT_TYPE").first()).group_by("EXIT_TYPE").len().to_dicts(),
                   dev_rows=dev.height, dev_failed_banks=nf(dev), test_rows=test.height, test_failed_banks=nf(test),
                   positive_rate_dev=dev["y"].mean(), positive_rate_test=test["y"].mean(),
                   rows_fail_before_avail=int(fin["fail_before_avail"].sum()), failure_only=FAILURE_ONLY)
    (REPORTS / "panel_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    assert summary["test_failed_banks"] >= 50, "test period must contain >= 50 failures"


if __name__ == "__main__":
    main()
