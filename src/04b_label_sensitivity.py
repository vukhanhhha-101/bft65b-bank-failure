"""PART 2c - How much does the label depend on the decision FAILURE_ONLY?
Counts positive rows and failed banks for three definitions of 'failure'. No model is trained.
Output: reports/label_sensitivity.csv and reports/assistance_continuity.csv"""
import polars as pl
from config import *

A = pl.col


def main():
    panel = pl.read_parquet(PANEL_CLEAN, columns=["CERT", "REPDTE", "qidx"])
    f = pl.read_parquet(PROC / "failures.parquet")
    defs = (("FAILURE only (used in the report)", f.filter(A("RESTYPE") == "FAILURE")),
            ("FAILURE + ASSISTANCE", f), ("ASSISTANCE only", f.filter(A("RESTYPE") == "ASSISTANCE")))
    rows = []
    for name, flt in defs:
        fd = flt.group_by("CERT").agg(A("FAILDATE").min())
        lab = (panel.join(fd, on="CERT", how="left").filter(A("REPDTE") <= LAST_LABEL_DATE)
                    .with_columns(((A("FAILDATE") > A("REPDTE")) & (A("FAILDATE") <= A("REPDTE").dt.offset_by("1y"))).fill_null(False).alias("y")))
        for period, cond in (("development", A("qidx") <= DEV_END_Q), ("test", A("qidx").is_between(TEST_START_Q, TEST_END_Q))):
            d = lab.filter(cond)
            rows.append(dict(definition=name, period=period, rows=d.height, positive_rows=int(d["y"].sum()),
                             positive_rate=float(d["y"].mean()), failed_banks=d.filter(A("y"))["CERT"].n_unique()))
    out = pl.DataFrame(rows); out.write_csv(REPORTS / "label_sensitivity.csv"); print(out)

    # do assisted banks keep reporting afterwards? (supports the decision not to call them failures)
    last = panel.group_by("CERT").agg(A("REPDTE").max().alias("last_report"))
    asst = f.filter(A("RESTYPE") == "ASSISTANCE").group_by("CERT").agg(A("FAILDATE").min()).join(last, on="CERT", how="inner")
    cont = pl.DataFrame([dict(assisted_banks_in_panel=asst.height,
                              still_reporting_after_assistance=asst.filter(A("last_report") > A("FAILDATE")).height,
                              median_quarters_reported_after=float(((asst.filter(A("last_report") > A("FAILDATE"))
                                    .select(((A("last_report") - A("FAILDATE")).dt.total_days() / 91.3).alias("q")))["q"]).median() or 0))])
    cont.write_csv(REPORTS / "assistance_continuity.csv"); print(cont)


if __name__ == "__main__":
    main()
