| layout | size_mb | write_s | q_2cols_all_years_s | q_one_year_all_cols_s |
|---|---|---|---|---|
| CSV | 446.4 | 13.86 | 2.533 | 2.866 |
| Parquet uncompressed | 395.4 | 6.05 | 0.074 | 0.049 |
| Parquet snappy | 209.6 | 1.63 | 0.076 | 0.386 |
| Parquet zstd-3 | 149.2 | 3.61 | 0.144 | 1.292 |
| Parquet zstd-3 partitioned by year (CHOSEN) | 156.3 | 4.31 | 0.094 | 0.027 |

Raw per-quarter API files (pandas default): 248.3 MB
