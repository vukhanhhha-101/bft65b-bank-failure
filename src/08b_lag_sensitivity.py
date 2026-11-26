import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.util
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score
from config import *

spec = importlib.util.spec_from_file_location("m08", Path(__file__).resolve().parent / "08_model.py")
m08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m08)

df = m08.load()
pc = pl.read_parquet(PANEL_CLEAN).select(["CERT", "REPDTE", "fail_before_avail"]).to_pandas()
df = df.merge(pc, on=["CERT", "REPDTE"], how="left")
dev = df[df.qidx <= DEV_END_Q].reset_index(drop=True)   # development period only; the test period is never read

oof = {m: np.full(len(dev), np.nan) for m in ["logit_all", "hgb_all"]}
for a, b in m08.FOLDS:
    va = (dev.qidx >= a) & (dev.qidx <= b)
    tr = dev.qidx <= a - m08.PURGE
    for m in oof:
        cols, mk = m08.MODELS[m]
        mod = mk().fit(dev.loc[tr, cols].values, dev.y[tr].values)
        oof[m][va.values] = mod.predict_proba(dev.loc[va, cols].values)[:, 1]

mask = ~np.isnan(oof["hgb_all"])
sub_all = dev[mask].copy()
for m in oof:
    sub_all[m] = oof[m][mask]
sub_no_lag = sub_all[~sub_all["fail_before_avail"].fillna(False)].copy()

rows = []
for sample_name, d in [("Dev OOF Full (1989-2006)", sub_all), ("Dev OOF Excl. <=45d Lag", sub_no_lag)]:
    y_s = d["y"].values
    for m in ["logit_all", "hgb_all"]:
        rows.append(dict(sample=sample_name, model=m, n_rows=len(d), n_pos=int(y_s.sum()),
                         pr_auc=round(float(average_precision_score(y_s, d[m].values)), 4),
                         roc_auc=round(float(roc_auc_score(y_s, d[m].values)), 4)))
out = pd.DataFrame(rows)
out.to_csv(REPORTS / "dev_lag_sensitivity.csv", index=False)
print("=== PUBLICATION-LAG (45 DAYS) SENSITIVITY ON DEV OOF ===")
print(out.to_string(index=False))
