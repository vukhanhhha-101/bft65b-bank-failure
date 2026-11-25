"""PART 5 - validation, baseline vs models, cost-based threshold, ONE test evaluation.
    python src/08_model.py --stage dev     # CV + threshold + fit on development period (test never touched)
    python src/08_model.py --stage test    # evaluate once on the test period (refuses to run twice)"""
import argparse, json
import joblib
import numpy as np, pandas as pd, polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import *

FEATS = json.loads((PROC / "feature_names.json").read_text())
BASE = ["asset_growth_4q", "roa", "npl_ratio", "equity_ratio"]
FOLDS = [(qi(1989, 1), qi(1991, 4)), (qi(1992, 1), qi(1994, 4)), (qi(1995, 1), qi(2000, 4)), (qi(2001, 1), qi(2006, 4))]
PURGE = 5   # train rows must be >= 5 quarters before the validation start: their 4-quarter label window must be fully in the past
FLAG = REPORTS / "TEST_USED.flag"


class Winsor(BaseEstimator, TransformerMixin):
    """Clip each feature at train-set quantiles (fitted on training data only)."""
    def __init__(self, q=0.005): self.q = q
    def fit(self, X, y=None):
        self.lo_, self.hi_ = np.nanquantile(X, self.q, axis=0), np.nanquantile(X, 1 - self.q, axis=0); return self
    def transform(self, X): return np.clip(X, self.lo_, self.hi_)


def logit():
    return Pipeline([("w", Winsor()), ("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("m", LogisticRegression(C=1.0, max_iter=500))])


def hgb():
    return Pipeline([("m", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_iter=200,
                                                          l2_regularization=1.0, random_state=SEED))])


class Raw:   # single-ratio rule: the Texas ratio itself is the risk score (no fitting)
    def fit(self, X, y): return self
    def predict_proba(self, X):
        s = np.nan_to_num(X[:, 0], nan=0.0); return np.c_[-s, s]


MODELS = {"rule_texas_ratio": (["texas_ratio"], Raw), "baseline_logit_4": (BASE, logit),
          "logit_all": (FEATS, logit), "hgb_all": (FEATS, hgb)}


def load():
    df = pl.read_parquet(FEATURES).to_pandas()
    df["FAILDATE"] = pd.to_datetime(df["FAILDATE"])
    return df


def cost_params(df_dev):
    f = pl.read_parquet(PROC / "failures.parquet")
    if FAILURE_ONLY: f = f.filter(pl.col("RESTYPE") == "FAILURE")
    f = f.filter(pl.col("FAILDATE") <= dt.date(DEV_END_Q // 4, 12, 31) + dt.timedelta(days=366))   # failures known in dev period only
    mean_cost = float(f["COST"].drop_nulls().mean())                                              # thousand USD per failed bank
    pos_rows = int(df_dev.y.sum()); banks = df_dev.loc[df_dev.y == 1, "CERT"].nunique()
    rows_per_failure = pos_rows / banks
    return dict(mean_cost_k=mean_cost, rows_per_failure=rows_per_failure, c_miss_row=mean_cost / rows_per_failure,
                c_fa_row=C_FALSE_ALARM_K, n_cost_obs=int(f["COST"].drop_nulls().len()))


def metrics(y, p, t=None, cp=None):
    out = dict(pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan, base_rate=y.mean())
    if t is not None:
        pred = p >= t; tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int(y.sum()) - tp
        out.update(recall=tp / max(1, y.sum()), precision=tp / max(1, pred.sum()), alarm_rate=pred.mean(), fp=fp, fn=fn,
                   cost_k=cp["c_miss_row"] * fn + cp["c_fa_row"] * fp,
                   cost_no_alarm_k=cp["c_miss_row"] * y.sum(), cost_all_alarm_k=cp["c_fa_row"] * (y == 0).sum())
    return out


def best_threshold(y, p, c_miss, c_fa):
    o = np.argsort(-p); ys = y[o]; tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    cost = np.r_[c_miss * P, c_miss * (P - tp) + c_fa * fp]            # k = 0 alarms, 1, 2, ...
    k = int(np.argmin(cost)); return (float(p[o][k - 1]) if k > 0 else float("inf")), k / len(y), float(cost[k])


def boot_ci(df, p, n=200):
    rng = np.random.default_rng(SEED); groups = df.groupby("CERT").indices; certs = np.array(list(groups)); y = df.y.values; res = []
    for _ in range(n):
        idx = np.concatenate([groups[c] for c in rng.choice(certs, len(certs))])
        if 0 < y[idx].sum(): res.append(average_precision_score(y[idx], p[idx]))
    return np.percentile(res, [2.5, 97.5]).tolist()


def stage_dev():
    df = load(); dev = df[df.pq <= DEV_END_Q].reset_index(drop=True)          # split on PREDICTION quarter
    assert TEST_START_Q - DEV_END_Q - 1 >= 4, "gap between development and test must be >= 4 quarters"
    cp = cost_params(dev); print("cost parameters:", cp)
    oof = {m: np.full(len(dev), np.nan) for m in MODELS}; rows = []
    for fi, (a, b) in enumerate(FOLDS, 1):
        va = (dev.pq >= a) & (dev.pq <= b); tr = dev.pq <= a - PURGE
        for m, (cols, mk) in MODELS.items():
            mod = mk().fit(dev.loc[tr, cols].values, dev.y[tr].values)
            p = mod.predict_proba(dev.loc[va, cols].values)[:, 1]; oof[m][va.values] = p
            rows.append(dict(fold=fi, model=m, n_val=int(va.sum()), n_pos=int(dev.y[va].sum()), **metrics(dev.y[va].values, p)))
            print(rows[-1])
    cv = pd.DataFrame(rows); cv.to_csv(REPORTS / "cv_scores.csv", index=False)
    mask = ~np.isnan(oof["hgb_all"]); y = dev.y.values[mask]
    pooled = {m: metrics(y, oof[m][mask]) for m in MODELS}; pd.DataFrame(pooled).T.to_csv(REPORTS / "cv_pooled.csv")
    print(pd.DataFrame(pooled).T)
    thr = {}
    for m in MODELS:
        t, ar, c = best_threshold(y, oof[m][mask], cp["c_miss_row"], cp["c_fa_row"]); thr[m] = dict(threshold=t, alarm_rate=ar, cv_cost_k=c)
    sens = []
    for cfa in (10, 25, 50, 100, 250, 500):
        for m in ("logit_all", "hgb_all"):
            t, ar, c = best_threshold(y, oof[m][mask], cp["c_miss_row"], cfa); sens.append(dict(c_fa_k=cfa, model=m, threshold=t, alarm_rate=ar))
    pd.DataFrame(sens).to_csv(REPORTS / "threshold_sensitivity.csv", index=False)
    final = {m: mk().fit(dev[cols].values, dev.y.values) for m, (cols, mk) in MODELS.items()}   # refit on all development data
    coefs = pd.Series(final["logit_all"].named_steps["m"].coef_[0], index=FEATS).sort_values()
    coefs.to_csv(REPORTS / "logit_coefficients.csv")
    joblib.dump(dict(models=final, thresholds=thr, cost=cp), PROC / "dev_artifacts.joblib")
    json.dump(dict(dev_rows=len(dev), dev_failed_banks=int(dev.loc[dev.y == 1, "CERT"].nunique()), thresholds=thr, cost=cp),
              open(REPORTS / "dev_summary.json", "w"), indent=1)
    print("thresholds:", thr, "\nDEV STAGE DONE - test untouched.")


def stage_test():
    if FLAG.exists():
        raise SystemExit("Test set already used once (reports/TEST_USED.flag). Re-running would be test-set reuse.")
    df = load(); test = df[(df.pq >= TEST_START_Q) & (df.pq <= TEST_END_Q)].reset_index(drop=True)
    art = joblib.load(PROC / "dev_artifacts.joblib"); cp = art["cost"]; rows = []
    print(f"TEST: {len(test):,} rows, {test.loc[test.y == 1, 'CERT'].nunique()} failed banks"); preds = {}
    for m, (cols, _) in MODELS.items():
        p = art["models"][m].predict_proba(test[cols].values)[:, 1]; preds[m] = p
        t = art["thresholds"].get(m, {}).get("threshold")
        r = dict(model=m, **metrics(test.y.values, p, t, cp) if t is not None else metrics(test.y.values, p))
        r["brier"] = np.nan if m == "rule_texas_ratio" else brier_score_loss(test.y.values, p)      # Texas ratio is a score, not a probability
        r["pr_auc_ci95"] = boot_ci(test, p); rows.append(r); print(r)
    pd.DataFrame(rows).to_csv(REPORTS / "test_results.csv", index=False)
    pd.DataFrame({"CERT": test.CERT.values, "REPDTE": test.REPDTE.values, "pq": test.pq.values, "y": test.y.values, **preds}).to_parquet(PROC / "test_predictions.parquet", index=False)
    FLAG.write_text("test evaluated once"); print("TEST STAGE DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["dev", "test"], required=True)
    a = ap.parse_args(); stage_dev() if a.stage == "dev" else stage_test()
