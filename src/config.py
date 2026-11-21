"""Shared settings and helpers. Every script does `from config import *`."""
from __future__ import annotations
import datetime as dt
import sys
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from fdic_download_starter import FIN_FIELDS  # field list used by the downloader (see README)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw" / "fdic"
PROC = ROOT / "data_proc"
REPORTS = ROOT / "reports"
PANEL = PROC / "panel"                 # optimised layout: Parquet, zstd, partitioned by year
PANEL_CLEAN = PROC / "panel_clean.parquet"
FEATURES = PROC / "features.parquet"
for p in (PROC, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
NUMERIC = [c for c in FIN_FIELDS if c not in ("CERT", "REPDTE", "STNAME")]
FLOW = ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR"]  # year-to-date flows in Call Reports

# ---- label / split design (justified in the report) ----
LAST_LABEL_DATE = dt.date(2025, 6, 30)   # later labels are incomplete
FAILURE_ONLY = True                      # keep RESTYPE == "FAILURE" (drop assistance transactions); sensitivity in report
def qi(year: int, q: int) -> int:        # quarter index
    return year * 4 + q - 1
DEV_END_Q = qi(2006, 4)                  # development: 1984Q1 - 2006Q4
TEST_START_Q, TEST_END_Q = qi(2009, 1), qi(2014, 4)   # test: 2009Q1 - 2014Q4 (gap = 8 quarters)
C_FALSE_ALARM_K = 50.0                   # ASSUMPTION: cost of one unnecessary review, thousand USD (sensitivity grid in 08)

QIDX = (pl.col("REPDTE").dt.year() * 4 + (pl.col("REPDTE").dt.month() - 1) // 3).alias("qidx")


def parse_date(c: str) -> pl.Expr:
    """FDIC dates come as YYYYMMDD, M/D/YYYY or YYYY-MM-DD depending on endpoint."""
    e = pl.col(c).cast(pl.Utf8)
    return pl.coalesce(e.str.to_date("%Y%m%d", strict=False),
                       e.str.to_date("%m/%d/%Y", strict=False),
                       e.str.to_date("%Y-%m-%d", strict=False)).alias(c)


def scan_panel() -> pl.LazyFrame:
    return pl.scan_parquet(str(PANEL / "year=*" / "*.parquet"), hive_partitioning=True)


def failure_dates() -> pl.DataFrame:
    """One row per failed CERT: first FAILDATE (+ COST)."""
    f = pl.read_parquet(PROC / "failures.parquet")
    if FAILURE_ONLY:
        f = f.filter(pl.col("RESTYPE") == "FAILURE")
    return f.group_by("CERT").agg(pl.col("FAILDATE").min(), pl.col("COST").first())


def md(df: pl.DataFrame, n: int = 40) -> str:
    d = df.head(n)
    out = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    for r in d.iter_rows():
        out.append("| " + " | ".join("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)
