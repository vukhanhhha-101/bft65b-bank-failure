"""PART 2d - reconcile FAILURE records, unique banks, clean panel and labelled rows."""
import pandas as pd

def funnel(f, raw_panel_certs, clean_panel_certs, feats, clean_exit, first_panel_date):
    fail = f[(f.RESTYPE == "FAILURE") & f.CERT.notna() & f.FAILDATE.notna()].copy()
    cert_fail = set(fail.CERT)
    in_raw = cert_fail & raw_panel_certs
    in_clean = cert_fail & clean_panel_certs
    multi = fail.groupby("CERT").size()
    first_fd = fail.groupby("CERT")["FAILDATE"].min()
    no_raw = sorted(cert_fail - raw_panel_certs)
    no_clean = sorted(cert_fail - clean_panel_certs)

    dev_pos = set(feats.loc[(feats.y == 1) & (feats.period == "development"), "CERT"])
    test_pos = set(feats.loc[(feats.y == 1) & (feats.period == "test"), "CERT"])
    any_pos = set(feats.loc[feats.y == 1, "CERT"])
    never_pos = sorted(in_clean - any_pos)

    rows = [
        ("Records in Failures table", len(f), "all RESTYPE values, including missing CERT"),
        ("  FAILURE records with valid CERT and FAILDATE", len(fail), "eligible for bank-level failure linkage"),
        ("  ASSISTANCE records", int((f.RESTYPE == "ASSISTANCE").sum()), "not failures under the primary definition"),
        ("Distinct CERT with a FAILURE record", len(cert_fail), "earliest FAILDATE used per CERT"),
        ("  CERT with multiple FAILURE records", int((multi > 1).sum()), "multiple event records retained"),
        ("FAILURE CERT with any raw Financials row", len(in_raw), "before ASSET validity and failure-date filters"),
        ("FAILURE CERT without raw Financials row", len(no_raw), "no matching raw Call Report CERT"),
        ("FAILURE CERT with at least one clean-panel row", len(in_clean), "after panel cleaning and point-in-time filters"),
        ("FAILURE CERT without clean-panel row", len(no_clean), "raw row may be invalid or all rows removed by event timing"),
        ("  No-panel first failure before first raw-panel quarter",
         int((first_fd.loc[no_raw] < first_panel_date).sum()) if no_raw else 0,
         "failure predates available raw Financials history"),
        ("Clean-panel banks with EXIT_TYPE=failure", int((clean_exit == "failure").sum()),
         "bank-level diagnostic classification"),
        ("Banks with y=1 in development", len(dev_pos), "at least one positive prediction row"),
        ("Banks with y=1 in test", len(test_pos), "at least one positive prediction row"),
        ("Banks with y=1 anywhere in modelling features", len(any_pos), "within configured modelling horizon"),
        ("Clean-panel FAILURE CERT never labelled y=1", len(never_pos),
         "e.g. event outside 12-month horizon or outside modelling period"),
    ]
    return pd.DataFrame(rows, columns=["step", "count", "explanation"])

def main():
    from config import (
        FEATURES, PANEL, PANEL_CLEAN, PROC, REPORTS,
        DEV_END_Q, TEST_START_Q, TEST_END_Q
    )
    f = pd.read_parquet(PROC / "failures.parquet", columns=["CERT", "FAILDATE", "RESTYPE"])
    f["FAILDATE"] = pd.to_datetime(f["FAILDATE"])
    raw = pd.read_parquet(str(PANEL), columns=["CERT", "REPDTE"])
    raw["REPDTE"] = pd.to_datetime(raw["REPDTE"])
    clean_panel = pd.read_parquet(PANEL_CLEAN, columns=["CERT", "EXIT_TYPE"])
    feats = pd.read_parquet(FEATURES, columns=["CERT", "pq", "y"])
    feats["period"] = "other"
    feats.loc[feats.pq <= DEV_END_Q, "period"] = "development"
    feats.loc[feats.pq.between(TEST_START_Q, TEST_END_Q), "period"] = "test"

    clean_exit = (
        clean_panel.drop_duplicates("CERT")
        .set_index("CERT")["EXIT_TYPE"]
    )
    out = funnel(
        f,
        set(raw["CERT"].dropna()),
        set(clean_panel["CERT"].dropna()),
        feats,
        clean_exit,
        raw["REPDTE"].min()
    )
    out.to_csv(REPORTS / "failure_funnel.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 90)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
