from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fdic_download_starter as st
from config import RAW, REPORTS


def api_total(endpoint: str) -> int:
    return int(
        st.get_json(
            endpoint,
            {"fields": "CERT", "limit": 1, "format": "json"}
        )["meta"]["total"]
    )


def inspect_failures(df: pd.DataFrame) -> dict:
    # 1. Missing CERT, classified by RESTYPE
    missing = df[df["CERT"].isna()]
    missing_by_restype = (
        missing["RESTYPE"]
        .fillna("MISSING")
        .value_counts()
        .to_dict()
    )

    # 2. CERT frequency, excluding missing identifiers
    valid = df[df["CERT"].notna()].copy()
    freq = valid.groupby("CERT").size()
    duplicate_groups = int((freq > 1).sum())
    duplicate_extra_rows = int((freq - 1).clip(lower=0).sum())

    # 3. Repeated event keys
    event_key_duplicates = int(
        valid.duplicated(
            subset=["CERT", "FAILDATE", "RESTYPE"]
        ).sum()
    )

    # 4. Exact duplicate rows across all columns
    exact_duplicates = int(df.duplicated().sum())

    # 5. Profile repeated CERT groups
    grouped = (
        valid.groupby("CERT")
        .agg(
            records=("CERT", "size"),
            distinct_faildates=("FAILDATE", "nunique"),
            distinct_restypes=("RESTYPE", "nunique"),
        )
    )
    repeated = grouped[grouped["records"] > 1]

    return {
        "rows": int(len(df)),
        "api_total": api_total("failures"),
        "rows_equal_api_total": int(len(df)) == api_total("failures"),
        "restype_counts": df["RESTYPE"].fillna("MISSING").value_counts().to_dict(),
        "null_cert": int(missing.shape[0]),
        "missing_cert_by_restype": {str(k): int(v) for k, v in missing_by_restype.items()},
        "duplicate_cert_groups": duplicate_groups,
        "duplicate_extra_rows": duplicate_extra_rows,
        "repeated_cert_groups_same_faildate": int((repeated["distinct_faildates"] == 1).sum()),
        "repeated_cert_groups_multiple_faildates": int((repeated["distinct_faildates"] > 1).sum()),
        "repeated_cert_groups_multiple_restypes": int((repeated["distinct_restypes"] > 1).sum()),
        "duplicate_event_key_rows": event_key_duplicates,
        "exact_duplicate_rows": exact_duplicates,
    }


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    df_fail = pd.read_parquet(RAW / "failures.parquet")
    df_inst = pd.read_parquet(RAW / "institutions.parquet")

    fail_diag = inspect_failures(df_fail)
    inst_total = api_total("institutions")
    fin_total = api_total("financials")

    out = {
        "failures": fail_diag,
        "institutions": {
            "rows": int(len(df_inst)),
            "api_total": inst_total,
            "rows_equal_api_total": int(len(df_inst)) == inst_total,
            "null_cert": int(df_inst["CERT"].isna().sum()),
            "duplicate_cert": int(df_inst["CERT"].dropna().duplicated().sum()),
        },
        "financials": {
            "api_total_now": fin_total,
        },
    }

    out_path = REPORTS / "small_tables_check.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(fail_diag, indent=2))


if __name__ == "__main__":
    main()
