"""POST-HOC DESCRIPTION of the single test evaluation (2009Q1-2014Q4).
The models, thresholds and features are the frozen development-period artifacts; nothing is tuned, selected
or changed here. It only adds descriptive numbers to the one test run:
  1. bank-level recall and lead time (how many quarters before the failure the first alarm appears)
  2. a paired bank-clustered bootstrap of PR-AUC(gradient boosting) - PR-AUC(logistic, 31 features)
Run as a script (python src/11_test_descriptives.py): the classes below must live in __main__ because the
saved scikit-learn pipelines were pickled from a script that defined them there."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import average_precision_score


class Winsor(BaseEstimator, TransformerMixin):          # same definition as in 08_model.py (needed to unpickle)
    def __init__(self, q=0.005): self.q = q
    def fit(self, X, y=None):
        self.lo_, self.hi_ = np.nanquantile(X, self.q, axis=0), np.nanquantile(X, 1 - self.q, axis=0); return self
    def transform(self, X): return np.clip(X, self.lo_, self.hi_)


class Raw:                                              # same definition as in 08_model.py (needed to unpickle)
    def fit(self, X, y): return self
    def predict_proba(self, X):
        s = np.nan_to_num(X[:, 0], nan=0.0); return np.c_[-s, s]


MODELS = ("logit_all", "hgb_all")


def bank_level(test, scores, thresholds):
    """test: rows with CERT, REPDTE, FAILDATE, y; scores: {model: array}. Returns one row per model."""
    pos_mask = test["y"].values == 1
    pos = test.loc[pos_mask].copy()
    pos["lead_months"] = (pd.to_datetime(pos["FAILDATE"]) - pd.to_datetime(pos["pred_date"])).dt.days / 30.44
    n_failed = pos["CERT"].nunique()
    nonfail_banks = test.loc[~test["CERT"].isin(pos["CERT"]), "CERT"].nunique()
    rows = []
    for m, s in scores.items():
        thr = thresholds[m]["threshold"]
        flagged = pos[s[pos_mask] >= thr]
        lead = flagged.groupby("CERT")["lead_months"].max()          # earliest alarm = longest lead
        nf_flagged = test.loc[(~test["CERT"].isin(pos["CERT"])) & (s >= thr), "CERT"].nunique()
        rows.append(dict(model=m, failed_banks=n_failed, caught_banks=int(flagged["CERT"].nunique()),
                         bank_recall=flagged["CERT"].nunique() / n_failed,
                         lead_median_months=float(lead.median()), lead_mean_months=float(lead.mean()),
                         share_lead_ge_6m=float((lead >= 6).mean()), share_lead_ge_9m=float((lead >= 9).mean()),
                         row_alarm_rate=float((s >= thr).mean()),
                         nonfailing_banks_flagged_at_least_once=int(nf_flagged),
                         share_of_nonfailing_banks=nf_flagged / max(1, nonfail_banks)))
    return pd.DataFrame(rows)


def paired_bootstrap(test, a, b, n_boot=1000, seed=42):
    """Bank-clustered bootstrap of PR-AUC(a) - PR-AUC(b)."""
    rng = np.random.default_rng(seed)
    groups = test.reset_index(drop=True).groupby("CERT").indices
    banks = np.array(list(groups)); y = test["y"].values
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(banks, len(banks))])
        if y[idx].sum():
            d.append(average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx]))
    d = np.array(d)
    return pd.DataFrame([dict(delta_pr_auc_mean=d.mean(), ci95_low=np.percentile(d, 2.5), ci95_high=np.percentile(d, 97.5),
                              share_of_resamples_gb_better=float((d > 0).mean()), n_boot=len(d))])


def main():
    from config import FEATURES, PROC, REPORTS, SEED, TEST_START_Q, TEST_END_Q
    art = joblib.load(PROC / "dev_artifacts.joblib")
    feats = json.loads((PROC / "feature_names.json").read_text())
    df = pd.read_parquet(FEATURES)
    test = df[(df.pq >= TEST_START_Q) & (df.pq <= TEST_END_Q)].reset_index(drop=True)   # same rule as 08_model.py
    X = test[feats].values.astype(float)
    scores = {m: art["models"][m].predict_proba(X)[:, 1] for m in MODELS}
    print(f"test rows {len(test):,} | positive rows {int(test.y.sum()):,} | failed banks {test.loc[test.y == 1, 'CERT'].nunique()}")
    bl = bank_level(test, scores, art["thresholds"]); bl.to_csv(REPORTS / "test_bank_level.csv", index=False)
    pb = paired_bootstrap(test, scores["hgb_all"], scores["logit_all"], seed=SEED); pb.to_csv(REPORTS / "test_paired_bootstrap.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(bl.T); print(pb.T)
    print("Post-hoc description of the single test evaluation; no model, threshold or feature was changed.")


if __name__ == "__main__":
    main()
