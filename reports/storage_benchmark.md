| layout | size_mb | write_s | q_2cols_all_years_s | q_one_year_all_cols_s |
|---|---|---|---|---|
| CSV | 446.4 | 9.87 | 3.408 | 3.469 |
| Parquet uncompressed | 395.4 | 4.04 | 0.097 | 0.386 |
| Parquet snappy | 209.6 | 3.73 | 0.102 | 0.702 |
| Parquet zstd-3 | 149.2 | 7.85 | 0.117 | 1.054 |
| Parquet zstd-3 partitioned by year (CHOSEN) | 156.3 | 8.07 | 0.296 | 0.086 |

Raw per-quarter API files (pandas default): 248.3 MB
