"""PART 3a - Polars (eager / lazy / streaming) vs pandas. Each run is a fresh process (clean memory numbers).
    python src/05_benchmark.py                 # full benchmark -> reports/benchmark.csv
    python src/05_benchmark.py --scales 1 4 8  # scale-up test (data replicated) to find the memory wall"""
import argparse, json, statistics, subprocess, sys, time
import psutil
from config import *

COLS = ["CERT", "REPDTE", "ASSET", "ROA", "NCLNLS"]
TASKS = ["agg", "window", "join"]
IMPLS = ["pandas", "polars_eager", "polars_lazy", "polars_streaming"]


def peak_mb():
    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 1e6      # Windows: true peak; Linux/macOS: current RSS (approximation)


# ---------- pandas ----------
def run_pandas(task, scale):
    import numpy as np, pandas as pd
    df = pd.read_parquet(str(PANEL), columns=COLS)
    df["REPDTE"] = pd.to_datetime(df["REPDTE"])
    if scale > 1:
        df = pd.concat([df.assign(CERT=df.CERT + i * 10_000_000) for i in range(scale)], ignore_index=True)
    if task == "agg":
        r = df.groupby(df.REPDTE.dt.year).agg(roa=("ROA", "mean"), assets=("ASSET", "sum"), n=("CERT", "size"))
        return float(r.loc[2008, "roa"])
    if task == "window":
        df = df.sort_values(["CERT", "REPDTE"]); df["q"] = df.REPDTE.dt.year * 4 + (df.REPDTE.dt.month - 1) // 3
        g = df.groupby("CERT")
        growth = np.where(g.q.shift(4) == df.q - 4, df.ASSET / g.ASSET.shift(4) - 1, np.nan)
        roa4 = g.ROA.rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)
        return float(np.nanmean(growth)) + 0 * float(roa4.mean())
    if task == "join":
        fd = failure_dates().to_pandas(); fd["FAILDATE"] = pd.to_datetime(fd["FAILDATE"])
        m = df.merge(fd[["CERT", "FAILDATE"]], on="CERT", how="left")
        y = (m.FAILDATE > m.REPDTE) & (m.FAILDATE <= m.REPDTE + pd.DateOffset(years=1))
        return float(y.sum())


# ---------- polars ----------
def pl_task(f, task):
    A = pl.col
    if task == "agg":
        return (f.group_by(A("REPDTE").dt.year().alias("yr")).agg(A("ROA").mean().alias("roa"), A("ASSET").sum(), pl.len())
                 .filter(A("yr") == 2008).select("roa"))
    if task == "window":
        f = f.sort(["CERT", "REPDTE"]).with_columns(QIDX)
        return f.select(pl.when(A("qidx").shift(4).over("CERT") == A("qidx") - 4)
                          .then(A("ASSET") / A("ASSET").shift(4).over("CERT") - 1).mean().alias("g"),
                        A("ROA").rolling_mean(4, min_samples=1).over("CERT").mean().alias("r"))
    if task == "join":
        fd = failure_dates().select(["CERT", "FAILDATE"]).lazy() if isinstance(f, pl.LazyFrame) else failure_dates().select(["CERT", "FAILDATE"])
        return (f.join(fd, on="CERT", how="left")
                 .select(((A("FAILDATE") > A("REPDTE")) & (A("FAILDATE") <= A("REPDTE").dt.offset_by("1y"))).sum().alias("y")))


def run_polars(impl, task, scale):
    path = str(PANEL / "year=*" / "*.parquet")
    if impl == "polars_eager":
        f = pl.read_parquet(path, columns=COLS)
        if scale > 1: f = pl.concat([f.with_columns(pl.col("CERT") + i * 10_000_000) for i in range(scale)])
        res = pl_task(f, task)
    else:
        lf = pl.scan_parquet(path).select(COLS)
        if scale > 1: lf = pl.concat([lf.with_columns(pl.col("CERT") + i * 10_000_000) for i in range(scale)])
        q = pl_task(lf, task)
        if task == "window" and scale == 1:
            (REPORTS / "polars_plan_optimized.txt").write_text(q.explain())
            (REPORTS / "polars_plan_naive.txt").write_text(q.explain(optimized=False))
        try:
            res = q.collect(engine="streaming") if impl == "polars_streaming" else q.collect()
        except TypeError:
            res = q.collect(streaming=impl == "polars_streaming")
    return float(res.row(0)[0])


def child(impl, task, scale):
    t = time.perf_counter()
    chk = run_pandas(task, scale) if impl == "pandas" else run_polars(impl, task, scale)
    print(json.dumps(dict(impl=impl, task=task, scale=scale, time_s=time.perf_counter() - t, peak_mb=peak_mb(), check=chk)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2); ap.add_argument("--scales", nargs="*", type=int, default=[1])
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if a.run:
        return child(a.run[0], a.run[1], a.scales[0])
    rows = []
    for s in a.scales:
        for task in TASKS:
            for impl in IMPLS:
                rs = []
                for _ in range(a.reps):
                    p = subprocess.run([sys.executable, __file__, "--run", impl, task, "--scales", str(s)], capture_output=True, text=True)
                    try: rs.append(json.loads(p.stdout.strip().splitlines()[-1]))
                    except Exception: print(p.stderr[-500:]); break
                if rs:
                    rows.append(dict(scale=s, task=task, impl=impl, time_median_s=round(statistics.median(r["time_s"] for r in rs), 2),
                                     time_min_s=round(min(r["time_s"] for r in rs), 2), peak_mb=round(max(r["peak_mb"] for r in rs)),
                                     check=round(rs[0]["check"], 6)))
                    print(rows[-1])
    out = pl.DataFrame(rows); out.write_csv(REPORTS / ("benchmark.csv" if a.scales == [1] else "benchmark_scaleup.csv"))
    print(out)


if __name__ == "__main__":
    main()
