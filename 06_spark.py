"""PART 3b - reproduce key steps in PySpark, check they match Polars, read the plan, test shuffle partitions.
Works locally (needs Java; on Windows use Google Colab / Databricks Free Edition if winutils is a problem)."""
import contextlib, io, time
import numpy as np
import polars as pl
from pyspark.sql import SparkSession, Window, functions as F
from config import *


def timed(fn):
    t = time.perf_counter(); r = fn(); return r, time.perf_counter() - t


def polars_side():
    A = pl.col
    lf = pl.scan_parquet(PANEL_CLEAN)
    by_year = (lf.group_by(A("REPDTE").dt.year().alias("yr")).agg(A("ROA").mean().alias("roa"), A("ASSET").sum().alias("assets"), pl.len().alias("n"))
                 .sort("yr").collect())
    growth = (lf.sort(["CERT", "REPDTE"]).with_columns(
                  pl.when(A("qidx").shift(4).over("CERT") == A("qidx") - 4).then(A("ASSET") / A("ASSET").shift(4).over("CERT") - 1).alias("g"))
                .group_by(A("REPDTE").dt.year().alias("yr")).agg(A("g").mean()).sort("yr").collect())
    y = lf.select(A("y").sum()).collect().item()
    return by_year, growth, y


def spark_side(spark):
    sdf = spark.read.parquet(str(PANEL_CLEAN))
    yr = F.year("REPDTE")
    by_year = sdf.groupBy(yr.alias("yr")).agg(F.avg("ROA").alias("roa"), F.sum("ASSET").alias("assets"), F.count("*").alias("n")).orderBy("yr")
    w = Window.partitionBy("CERT").orderBy("qidx")
    g = sdf.withColumn("g", F.when(F.lag("qidx", 4).over(w) == F.col("qidx") - 4, F.col("ASSET") / F.lag("ASSET", 4).over(w) - 1))
    growth = g.groupBy(yr.alias("yr")).agg(F.avg("g").alias("g")).orderBy("yr")
    fd = spark.createDataFrame(failure_dates().select(["CERT", "FAILDATE"]).to_pandas())
    ylab = sdf.drop("y").join(fd.withColumnRenamed("FAILDATE", "FD"), on="CERT", how="left").select(
        F.sum(((F.col("FD") > F.col("REPDTE")) & (F.col("FD") <= F.add_months(F.col("REPDTE"), 12))).cast("int")).alias("y"))
    return by_year, growth, ylab


def main():
    pl_year, pl_growth, pl_y = polars_side()
    res = []
    for parts in (200, 16):
        spark = (SparkSession.builder.master("local[*]").appName("bft65b").config("spark.sql.shuffle.partitions", str(parts))
                 .config("spark.driver.memory", "4g").getOrCreate())
        by_year, growth, ylab = spark_side(spark)
        (sy, t1), (sg, t2), (sl, t3) = timed(lambda: by_year.toPandas()), timed(lambda: growth.toPandas()), timed(lambda: ylab.collect()[0][0])
        res.append(dict(shuffle_partitions=parts, agg_s=round(t1, 2), window_s=round(t2, 2), join_s=round(t3, 2)))
        if parts == 16:
            ok1 = np.allclose(sy["roa"], pl_year["roa"].to_numpy(), rtol=1e-9, equal_nan=True) and (sy["n"].to_numpy() == pl_year["n"].to_numpy()).all()
            ok2 = np.allclose(sg["g"].to_numpy(dtype=float), pl_growth["g"].to_numpy(), rtol=1e-6, equal_nan=True)
            ok3 = int(sl) == int(pl_y)
            print(f"MATCH  agg={ok1}  window={ok2}  label_sum={ok3} (spark {sl} vs polars {pl_y})")
            (REPORTS / "spark_match.txt").write_text(f"agg={ok1}\nwindow={ok2}\nlabel_sum={ok3}\nspark_y={sl}\npolars_y={pl_y}\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                growth.explain("formatted")
            (REPORTS / "spark_plan_window.txt").write_text(buf.getvalue())
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                by_year.explain("formatted")
            (REPORTS / "spark_plan_agg.txt").write_text(buf.getvalue())
        spark.stop()
    out = pl.DataFrame(res); out.write_csv(REPORTS / "spark_timing.csv"); print(out)


if __name__ == "__main__":
    main()
