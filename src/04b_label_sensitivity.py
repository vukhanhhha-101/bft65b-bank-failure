"""PART 2c - label sensitivity under alternative event definitions.

All definitions use the same point-in-time prediction date and 12-month
outcome horizon as 04_panel.py. No model is trained.
"""
import polars as pl
from config import *

A = pl.col

def first_event(events: pl.DataFrame, event_name: str) -> pl.DataFrame:
    return (
        events.filter(
            (A("RESTYPE") == event_name)
            & A("CERT").is_not_null()
            & A("FAILDATE").is_not_null()
        )
        .group_by("CERT")
        .agg(A("FAILDATE").min().alias("event_date"))
    )

def main():
    panel = pl.read_parquet(
        PANEL_CLEAN,
        columns=["CERT", "REPDTE", "qidx", "pq", "pred_date", "FAILDATE"]
    ).filter(
        (A("REPDTE") <= pl.lit(LAST_REPORT_DATE))
        & (A("pred_date") <= pl.lit(LAST_LABEL_DATE))
    )
    events = pl.read_parquet(PROC / "failures.parquet")
    fail_date = first_event(events, "FAILURE").rename({"event_date": "first_failure"})
    assist_date = first_event(events, "ASSISTANCE").rename({"event_date": "first_assistance"})

    # A bank-level event table; assistance after an earlier failure is not a valid
    # assistance outcome because the bank had already failed.
    event_table = fail_date.join(assist_date, on="CERT", how="full", coalesce=True)

    definitions = []
    definitions.append((
        "FAILURE only (primary definition)",
        fail_date.select(["CERT", pl.col("first_failure").alias("event_date")])
    ))

    combined = event_table.with_columns(
        pl.min_horizontal("first_failure", "first_assistance").alias("event_date")
    ).select(["CERT", "event_date"])
    definitions.append(("FAILURE + ASSISTANCE", combined))

    assistance_only = (
        event_table
        .filter(
            A("first_assistance").is_not_null()
            & (
                A("first_failure").is_null()
                | (A("first_assistance") < A("first_failure"))
            )
        )
        .select(["CERT", pl.col("first_assistance").alias("event_date")])
    )
    definitions.append(("ASSISTANCE only (before any FAILURE)", assistance_only))

    rows = []
    for name, event_dates in definitions:
        lab = (
            panel.join(event_dates, on="CERT", how="left")
            # If the event already occurred by prediction date, the bank is no longer
            # in the risk set for this definition.
            .filter(A("event_date").is_null() | (A("event_date") > A("pred_date")))
            .with_columns(
                (
                    A("event_date").is_not_null()
                    & (A("event_date") > A("pred_date"))
                    & (A("event_date") <= A("pred_date").dt.offset_by("1y"))
                ).fill_null(False).cast(pl.Int8).alias("y_sensitivity")
            )
        )

        for period, condition in (
            ("development", A("pq") <= DEV_END_Q),
            ("test", A("pq").is_between(TEST_START_Q, TEST_END_Q)),
        ):
            d = lab.filter(condition)
            rows.append({
                "definition": name,
                "period": period,
                "rows_at_risk": d.height,
                "positive_rows": int(d["y_sensitivity"].sum() or 0),
                "positive_rate": float(d["y_sensitivity"].mean() or 0.0),
                "positive_banks": d.filter(A("y_sensitivity") == 1)["CERT"].n_unique()
            })

    out = pl.DataFrame(rows)
    out.write_csv(REPORTS / "label_sensitivity.csv")
    print(out)

    # Continuity after assistance: descriptive only, not a causal conclusion.
    last_report = panel.group_by("CERT").agg(A("REPDTE").max().alias("last_report"))
    assistance = assist_date.join(last_report, on="CERT", how="inner")
    after = assistance.filter(A("last_report") > A("first_assistance"))
    quarters = (
        (after["last_report"] - after["first_assistance"])
        .dt.total_days() / 91.3
    )
    continuity = pl.DataFrame([{
        "assisted_banks_in_panel": assistance.height,
        "still_reporting_after_assistance": after.height,
        "median_quarters_reported_after_assistance": float(quarters.median() or 0.0)
    }])
    continuity.write_csv(REPORTS / "assistance_continuity.csv")
    print(continuity)

if __name__ == "__main__":
    main()
