"""PART 1 - normalise schema, benchmark storage layouts, build the optimised panel."""
import shutil, time
import polars as pl
from config import *


def norm_fin(f: Path) -> pl.DataFrame:
    df = pl.read_parquet(f)
    for c in FIN_FIELDS:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))
    return df.select(FIN_FIELDS).with_columns(
        pl.col("CERT").cast(pl.Int32), parse_date("REPDTE"), pl.col("STNAME").cast(pl.Utf8),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in NUMERIC])


def size_mb(p: Path) -> float:
    p = Path(p)
    return (sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) if p.is_dir() else p.stat().st_size) / 1e6


def best_of(fn, n=3):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)


def main():
    files = sorted((RAW / "financials").glob("REPDTE=*.parquet"))
    df = pl.concat([norm_fin(f) for f in files]).sort(["CERT", "REPDTE"])
    print(f"{len(files)} quarter files -> {df.height:,} rows, in-memory {df.estimated_size('mb'):.0f} MB")
    bench = PROC / "bench"; bench.mkdir(exist_ok=True)

    # small tables (normalised)
    fl = pl.read_parquet(RAW / "failures.parquet")
    fl.with_columns(pl.col("CERT").cast(pl.Int32), parse_date("FAILDATE"),
                    pl.col("COST").cast(pl.Float64, strict=False)).write_parquet(PROC / "failures.parquet", compression="zstd")
    ins = pl.read_parquet(RAW / "institutions.parquet")
    ins.with_columns(pl.col("CERT").cast(pl.Int32), parse_date("ESTYMD"), parse_date("ENDEFYMD"), parse_date("INSDATE"),
                     pl.col("CHANGEC1").cast(pl.Utf8), pl.col("ACTIVE").cast(pl.Int8, strict=False)
                     ).write_parquet(PROC / "institutions.parquet", compression="zstd")

    rows = []
    def rec(name, path, w, scan, part=False):
        yr = (pl.col("year") == 2008) if part else (pl.col("REPDTE").dt.year() == 2008)
        q_agg = best_of(lambda: scan().group_by(pl.col("REPDTE").dt.year().alias("yr")).agg(pl.col("ROA").mean()).collect())
        q_one = best_of(lambda: scan().filter(yr).collect())
        rows.append(dict(layout=name, size_mb=round(size_mb(path), 1), write_s=round(w, 2),
                         q_2cols_all_years_s=round(q_agg, 3), q_one_year_all_cols_s=round(q_one, 3)))
        print(rows[-1])

    t = time.perf_counter(); df.write_csv(bench / "panel.csv"); w = time.perf_counter() - t
    rec("CSV", bench / "panel.csv", w, lambda: pl.scan_csv(bench / "panel.csv", try_parse_dates=True))
    for name, comp, lvl in (("Parquet uncompressed", "uncompressed", None), ("Parquet snappy", "snappy", None),
                            ("Parquet zstd-3", "zstd", 3)):
        p = bench / f"{comp}.parquet"
        t = time.perf_counter(); df.write_parquet(p, compression=comp, compression_level=lvl); w = time.perf_counter() - t
        rec(name, p, w, lambda p=p: pl.scan_parquet(p))

    if PANEL.exists():
        shutil.rmtree(PANEL)
    t = time.perf_counter()
    for y, g in df.with_columns(pl.col("REPDTE").dt.year().alias("year")).partition_by("year", as_dict=True).items():
        d = PANEL / f"year={y[0]}"; d.mkdir(parents=True)
        g.drop("year").write_parquet(d / "part.parquet", compression="zstd", compression_level=3)
    w = time.perf_counter() - t
    rec("Parquet zstd-3 partitioned by year (CHOSEN)", PANEL, w, scan_panel, part=True)

    raw_mb = size_mb(RAW / "financials")
    out = pl.DataFrame(rows)
    out.write_csv(REPORTS / "storage_benchmark.csv")
    (REPORTS / "storage_benchmark.md").write_text(md(out) + f"\n\nRaw per-quarter API files (pandas default): {raw_mb:.1f} MB\n")
    print(out)


if __name__ == "__main__":
    main()
