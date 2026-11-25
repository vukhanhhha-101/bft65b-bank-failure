"""PART 3c - Spark vs Polars scale-up on the window task (same input file, data replicated x1, x4, x8).
Each engine reads the clean panel from disk inside the timed section; Spark gets one warm-up run first
(JVM start-up and code generation) and 16 shuffle partitions (best setting found in 06_spark.py).
    python src/06b_spark_scaleup.py [--scales 1 4 8] [--reps 3]
Output: reports/spark_scaleup.csv"""
import argparse, statistics, time
from functools import reduce
import numpy as np
import polars as pl
from pyspark.sql import SparkSession, Window, functions as F
from config import *

OFFSET = 10_000_000          # makes the replicated banks distinct


def polars_window(scale):
    lf = pl.scan_parquet(PANEL_CLEAN).select(["CERT", "ASSET", "qidx"])
    if scale > 1:
        lf = pl.concat([lf.with_columns(pl.col("CERT") + i * OFFSET) for i in range(scale)])
    q = (lf.sort(["CERT", "qidx"])
           .select(pl.when(pl.col("qidx").shift(4).over("CERT") == pl.col("qidx") - 4)
                     .then(pl.col("ASSET") / pl.col("ASSET").shift(4).over("CERT") - 1).mean().alias("g")))
    return float(q.collect().item())


def spark_window(spark, scale):
    sdf = spark.read.parquet(str(PANEL_CLEAN)).select("CERT", "ASSET", "qidx")
    if scale > 1:
        sdf = reduce(lambda a, b: a.unionByName(b), [sdf.withColumn("CERT", F.col("CERT") + i * OFFSET) for i in range(scale)])
    w = Window.partitionBy("CERT").orderBy("qidx")
    g = F.when(F.lag("qidx", 4).over(w) == F.col("qidx") - 4, F.col("ASSET") / F.lag("ASSET", 4).over(w) - 1)
    return float(sdf.withColumn("g", g).select(F.avg("g")).collect()[0][0])


def timed(fn, reps):
    ts, res = [], None
    for _ in range(reps):
        t = time.perf_counter(); res = fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts), min(ts), res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scales", nargs="*", type=int, default=[1, 4, 8]); ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    n = pl.scan_parquet(PANEL_CLEAN).select(pl.len()).collect().item()
    spark = (SparkSession.builder.master("local[*]").appName("bft65b-scaleup").config("spark.sql.shuffle.partitions", "16")
             .config("spark.driver.memory", "6g").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    spark_window(spark, 1)                                           # warm-up, not timed
    rows = []
    for s in a.scales:
        tp, mp, rp = timed(lambda: polars_window(s), a.reps)
        ts_, ms, rs = timed(lambda: spark_window(spark, s), a.reps)
        match = bool(np.isclose(rp, rs, rtol=1e-6))
        rows += [dict(engine="polars_lazy", scale=s, rows=n * s, time_median_s=round(tp, 2), time_min_s=round(mp, 2), check=rp, match=match),
                 dict(engine="spark_local", scale=s, rows=n * s, time_median_s=round(ts_, 2), time_min_s=round(ms, 2), check=rs, match=match)]
        print(rows[-2]); print(rows[-1], flush=True)
    out = pl.DataFrame(rows); out.write_csv(REPORTS / "spark_scaleup.csv"); print(out)
    spark.stop()


if __name__ == "__main__":
    main()
