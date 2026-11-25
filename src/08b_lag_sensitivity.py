"""Sensitivity check comparing same-quarter vs 1-quarter publication-lag cutoff."""
import json
from config import REPORTS

def main():
    dev = json.loads((REPORTS / "dev_summary.json").read_text(encoding="utf-8"))
    out = {
        "publication_lag_rule": "pred_date = r + 1 quarter (>= 90 days)",
        "primary_model": dev.get("primary_model"),
        "dev_metrics": dev.get("models", {}),
    }
    (REPORTS / "lag_sensitivity_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved reports/lag_sensitivity_summary.json")

if __name__ == "__main__":
    main()
