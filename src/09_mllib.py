"""PART 5b - short Spark MLlib demo. Trains on 1984-1999, validates on 2001-2006 (DEVELOPMENT data only; test untouched)."""
import json
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.sql import SparkSession, functions as F
from config import *

FEATS = json.loads((PROC / "feature_names.json").read_text())


def main():
    spark = SparkSession.builder.master("local[*]").appName("bft65b-mllib").config("spark.driver.memory", "4g").getOrCreate()
    sdf = spark.read.parquet(str(FEATURES)).withColumn("label", F.col("y").cast("double"))
    for c in FEATS: sdf = sdf.withColumn(c, F.col(c).cast("double"))
    tr = sdf.filter(F.col("qidx") <= qi(1999, 4)); va = sdf.filter(F.col("qidx").between(qi(2001, 1), qi(2006, 4)))
    imp = Imputer(strategy="median", inputCols=FEATS, outputCols=[f + "_i" for f in FEATS])
    vec = VectorAssembler(inputCols=[f + "_i" for f in FEATS], outputCol="raw")
    sc = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True)
    ev_pr, ev_roc = BinaryClassificationEvaluator(metricName="areaUnderPR"), BinaryClassificationEvaluator(metricName="areaUnderROC")
    rows = []
    for name, est in (("MLlib logistic", LogisticRegression(maxIter=100, regParam=0.01)),
                      ("MLlib GBT", GBTClassifier(maxIter=50, maxDepth=4, seed=SEED))):
        pred = Pipeline(stages=[imp, vec, sc, est]).fit(tr).transform(va)
        rows.append(dict(model=name, pr_auc=ev_pr.evaluate(pred), roc_auc=ev_roc.evaluate(pred)))
        print(rows[-1])
    out = pd.DataFrame(rows); out.to_csv(REPORTS / "mllib_demo.csv", index=False)
    try:
        cv = pd.read_csv(REPORTS / "cv_scores.csv"); print("sklearn fold-4 (val 2001-2006, train <= 2000) for comparison:\n", cv[cv.fold == 4][["model", "pr_auc", "roc_auc"]])
    except FileNotFoundError:
        pass
    spark.stop()


if __name__ == "__main__":
    main()
