"""PART 4a - point-in-time feature table. Only values at quarter q and EARLIER quarters of the same bank are used."""
import json
import numpy as np
import polars as pl
from config import *

A = pl.col


def sdiv(n, d):                     # ratio only when the denominator is positive, else null (never inf)
    return pl.when(d > 0).then(n / d)


def lagq(c, k):                     # value k QUARTERS ago (null if the bank has a gap) - uses the past only
    return pl.when(A("qidx").shift(k).over("CERT") == A("qidx") - k).then(A(c).shift(k).over("CERT"))


L1 = {   # level-1: ratios at quarter q  (CAMELS: Capital, Asset quality, Management, Earnings, Liquidity, Sensitivity/size)
    "log_asset": A("ASSET").log(),
    "equity_ratio": sdiv(A("EQ"), A("ASSET")),
    "lev_ratio_reg": A("RBC1AAJ"), "rbc_total": A("RBCRWAJ"),                      # verify meaning in the field dictionary
    "loan_ratio": sdiv(A("LNLSGR"), A("ASSET")),
    "re_share": sdiv(A("LNRE"), A("LNLSGR")), "ci_share": sdiv(A("LNCI"), A("LNLSGR")), "con_share": sdiv(A("LNCON"), A("LNLSGR")),
    "npl_ratio": sdiv(A("NCLNLS"), A("LNLSGR")),
    "pd30_ratio": sdiv(A("P3ASSET"), A("ASSET")), "pd90_ratio": sdiv(A("P9ASSET"), A("ASSET")),
    "nonaccrual_ratio": sdiv(A("NAASSET"), A("ASSET")), "ore_ratio": sdiv(A("ORE"), A("ASSET")),
    "reserve_ratio": sdiv(A("LNATRES"), A("LNLSGR")), "reserve_cover": sdiv(A("LNATRES"), A("NCLNLS")),
    "texas_ratio": sdiv(A("NCLNLS") + A("ORE"), A("EQ") + A("LNATRES")),
    "liquidity": sdiv(A("CHBAL") + A("SC"), A("ASSET")),
    "brokered_ratio": sdiv(A("BRO"), A("DEP")), "deposit_ratio": sdiv(A("DEP"), A("ASSET")),
    "roa": A("ROA"), "roe": A("ROE"), "nim": A("NIMY"),
    "roa_q": sdiv(4 * A("NETINC_Q"), A("ASSET")),                                  # quarterly (de-cumulated), annualised
    "provision_ratio": sdiv(4 * A("ELNATR_Q"), A("ASSET")),
    "cost_income": sdiv(A("NONIX_Q"), A("INTINC_Q") - A("EINTEXP_Q") + A("NONII_Q")),
}
BASE = ["asset_growth_4q", "roa", "npl_ratio", "equity_ratio"]


def build(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.sort(["CERT", "REPDTE"]).with_columns([e.alias(k) for k, e in L1.items()])
    L2 = {
        "asset_growth_4q": pl.when(lagq("ASSET", 4) > 0).then(A("ASSET") / lagq("ASSET", 4) - 1),
        "roa_chg_4q": A("roa") - lagq("roa", 4),
        "npl_chg_4q": A("npl_ratio") - lagq("npl_ratio", 4),
        "equity_chg_4q": A("equity_ratio") - lagq("equity_ratio", 4),
        "roa_mean_4q": pl.mean_horizontal(A("roa"), lagq("roa", 1), lagq("roa", 2), lagq("roa", 3)),
        "loan_growth_4q": pl.when(lagq("LNLSGR", 4) > 0).then(A("LNLSGR") / lagq("LNLSGR", 4) - 1),
    }
    return lf.with_columns([e.alias(k) for k, e in L2.items()])


FEATS = list(L1) + ["asset_growth_4q", "roa_chg_4q", "npl_chg_4q", "equity_chg_4q", "roa_mean_4q", "loan_growth_4q"]


def main():
    lf = pl.scan_parquet(PANEL_CLEAN).filter(A("REPDTE") <= LAST_LABEL_DATE)
    full = build(lf).select(["CERT", "REPDTE", "qidx", "y", "FAILDATE"] + FEATS).collect()
    full = full.with_columns([pl.when(A(c).is_infinite()).then(None).otherwise(A(c)).alias(c) for c in FEATS])
    full.write_parquet(FEATURES, compression="zstd")
    (PROC / "feature_names.json").write_text(json.dumps(FEATS))

    # ---- point-in-time / look-ahead test ----
    # recompute the features using ONLY data up to a cut-off quarter and compare with the full-history table at that quarter.
    for cut in (dt.date(1995, 12, 31), dt.date(2005, 12, 31), dt.date(2012, 12, 31)):
        part = build(lf.filter(A("REPDTE") <= cut)).filter(A("REPDTE") == cut).select(["CERT"] + FEATS).collect().sort("CERT")
        ref = full.filter(A("REPDTE") == cut).select(["CERT"] + FEATS).sort("CERT")
        assert part["CERT"].to_list() == ref["CERT"].to_list()
        for c in FEATS:
            a, b = part[c].to_numpy().astype(float), ref[c].to_numpy().astype(float)
            assert np.allclose(a, b, equal_nan=True, rtol=1e-9), f"LOOK-AHEAD in {c} at {cut}"
    msg = f"point-in-time test passed for 3 cut-offs; {full.height:,} rows x {len(FEATS)} features"
    (REPORTS / "pit_test.txt").write_text(msg); print(msg)
    print(full.select(FEATS).null_count().transpose(include_header=True).head(40))


if __name__ == "__main__":
    main()
