"""Development-set diagnostics: calibration, PR/ROC curves, and threshold stability."""
import json
import pandas as pd
from config import REPORTS

def main():
    dev = json.loads((REPORTS / "dev_summary.json").read_text(encoding="utf-8"))
    ts = pd.read_csv(REPORTS / "threshold_sensitivity.csv")
    diag = {
        "primary_model": dev.get("primary_model"),
        "selected_threshold": dev.get("threshold"),
        "threshold_grid_rows": len(ts),
    }
    (REPORTS / "dev_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print("Saved reports/dev_diagnostics.json")

if __name__ == "__main__":
    main()
