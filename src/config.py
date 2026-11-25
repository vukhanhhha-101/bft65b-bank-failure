"""Shared settings and helpers. Every script does `from config import *`."""
from __future__ import annotations
import datetime as dt
import sys
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from fdic_download_starter import FIN_FIELDS

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw" / "fdic"
PROC = ROOT / "data_proc"
REPORTS = ROOT / "reports"
PANEL = PROC / "panel"
PANEL_CLEAN = PROC / "panel_clean.parquet"
FEATURES = PROC / "features.parquet"
for p in (PROC, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
NUMERIC = [c for c in FIN_FIELDS if c not in ("CERT", "REPDTE", "STNAME")]
FLOW = ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR"]

# Point-in-time design:
# A report dated r is used at the end of the following quarter.
# The prediction window is (pred_date, pred_date + 1 year].
PUB_LAG_MONTHS = 3
MIN_PUB_LAG_DAYS = 45
LAST_LABEL_DATE = dt.date(2025, 6, 30)   # final prediction date (2025Q2)
LAST_REPORT_DATE = dt.date(2025, 3, 31)  # final report date usable at 2025Q2
FAILURE_ONLY = True

def qi(year: int, q: int) -> int:
    return year * 4 + q - 1

DEV_END_Q = qi(2006, 4)                 # prediction quarters
TEST_START_Q, TEST_END_Q = qi(2009, 1), qi(2014, 4)
C_FALSE_ALARM_K = 50.0

QIDX = (pl.col("REPDTE").dt.year() * 4 + (pl.col("REPDTE").dt.month() - 1) // 3).alias("qidx")

def parse_date(c: str) -> pl.Expr:
    e = pl.col(c).cast(pl.Utf8)
    return pl.coalesce(
        e.str.to_date("%Y%m%d", strict=False),
        e.str.to_date("%m/%d/%Y", strict=False),
        e.str.to_date("%Y-%m-%d", strict=False)
    ).alias(c)

def scan_panel() -> pl.LazyFrame:
    return pl.scan_parquet(str(PANEL / "year=*" / "*.parquet"), hive_partitioning=True)

def failure_dates() -> pl.DataFrame:
    """One row per valid CERT: earliest event date and non-null COST."""
    f = pl.read_parquet(PROC / "failures.parquet")

    if FAILURE_ONLY:
        f = f.filter(pl.col("RESTYPE") == "FAILURE")

    f = f.filter(
        pl.col("CERT").is_not_null()
        & pl.col("FAILDATE").is_not_null()
    )

    return f.group_by("CERT").agg(
        pl.col("FAILDATE").min().alias("FAILDATE"),
        pl.col("COST").drop_nulls().first().alias("COST")
    )
def md(df: pl.DataFrame, n: int = 40) -> str:
    d = df.head(n)
    out = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    for r in d.iter_rows():
        out.append("| " + " | ".join(
            "" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
            for v in r
        ) + " |")
    return "\n".join(out)
