# Scientific Methodology: Precipitation Trends

## Purpose and analysis unit

The engine estimates monotonic precipitation change independently for each grid cell and requested period. Statistical significance describes evidence against a no-monotonic-trend null; it does not establish cause, forecast future rainfall, or measure environmental importance. Preserve the source product, parameter, units, grid coordinates, retrieval/version metadata, and analysis settings with every result.

## Processing and data quality

1. Validate numeric, finite precipitation amounts; reject negative values, duplicate timestamps, unsupported cadence, malformed coordinates, and ambiguous units. Treat zero as a valid observation. Convert declared source missing-value sentinels to missing values; do not infer that an absent record means zero.
2. Sort observations by timestamp and align them to the declared calendar/cadence. Report gaps, duplicates, invalid values, and per-period coverage. Do not impute missing precipitation for trend testing in the MVP: imputation can create or suppress trends and complicate uncertainty. Keep source missingness distinct from quality-control rejection.
3. Sum daily amounts into calendar months and monthly amounts into calendar years. Do not sum rates without first converting them to amounts. Only emit an aggregate when its configured coverage rule is met; otherwise mark it missing. The rule (for example, required fraction of expected daily observations) is a documented analysis setting, not a universal scientific constant. Do not treat incomplete first/last years as complete annual totals.

## Trend methods

| Method | Calculation and output | Why / assumptions | Limitations |
|---|---|---|---|
| Seasonal Mann–Kendall (monthly series) | For each calendar month, compare available pairs of years and sum signed ranks across months. Use this seasonal statistic with a moving year-block bootstrap of trend-free residuals for the two-sided p-value. Report statistic, p-value, season counts, block length, resamples, and seed. | Controls the strong annual rainfall cycle by comparing like months. Resampling whole years retains the month pattern and within-year dependence; consecutive year blocks retain short-range between-year dependence. | Assumes residual blocks are representative of the no-trend process. It can be unreliable for short records, long persistence, or unstable variance. It tests a common monotonic trend, not seasonal timing changes or arbitrary nonlinear/step patterns. |
| Mann–Kendall (annual series) | Compute S, tie-corrected variance, standardized statistic, and two-sided p-value using the Hamed–Rao rank-autocorrelation variance correction for complete annual records. If annual periods are missing, use the moving year-block bootstrap p-value instead. Report method, variance adjustment or block settings, and seed. | A rank test for monotonic change that does not require normally distributed totals; annual aggregation reduces the within-year seasonal cycle. The variance correction addresses serial autocorrelation. | The test detects monotonic association, not cause or magnitude. The asymptotic adjusted p-value can be unreliable for short records, many ties, or poorly estimated dependence. Block bootstrap results depend on block choice and record length. |
| Sen slope | For annual data, take the median of all pairwise precipitation differences divided by elapsed years. For monthly seasonal results, form slopes only between observations from the same calendar month, then take their median. Return slope in source units per year. | Robust, interpretable magnitude estimate that aligns with rank-based trend analysis and tolerates skew and outliers better than least-squares slope. | Summarizes a roughly monotonic linear rate; it can hide nonlinear behavior or season-specific trends. Monthly seasonal slope describes a common within-month rate, not a seasonal-cycle change. |
| Slope confidence interval | Return a percentile interval at the requested confidence level from moving year-block resamples of detrended residuals. For monthly data, resample complete year rows and restore month baselines; for annual data, resample contiguous annual residual blocks. Record block length, replicate count, confidence level, and random seed. | Quantifies uncertainty in the Sen rate while retaining dependence better than an independent-observation interval. | Coverage is approximate and depends on block length, record length, and stable residual behavior. The current implementation can return no interval if no valid bootstrap slopes are produced. The interval is not a confidence interval for environmental impact. |

The seasonal and annual analysis are separate products: monthly input can produce a seasonal monthly trend and, subject to coverage, annual totals and an annual trend. The seasonal method is the default for monthly trends. Do not use ordinary MK across raw monthly values as the primary monthly test. The Hamed–Rao adjustment is used for complete annual series; monthly series use a seasonal statistic with year-block bootstrap inference. Neither procedure removes the need to inspect persistence, coverage, and residual behavior.

## Missing data and eligibility

Never fill missing observations silently. Pairwise seasonal comparisons allow unequal available counts, but report those counts and assess whether missingness is extensive or systematically concentrated in particular years/seasons. Aggregates failing the coverage rule remain missing. Define minimum record length, per-season coverage, and minimum usable comparisons as versioned configuration. These are operational eligibility rules, not universal properties of the MK test. If a cell fails them, label it **insufficient data** and provide the failing criteria; do not return a p-value or imply no trend.

## Spatial testing and multiplicity

Run the selected trend method independently for every eligible grid cell. Define a testing family before examining results (normally all eligible cells for one variable, cadence, and analysis period). Apply Benjamini–Yekutieli false discovery rate (FDR) adjustment to the family’s valid p-values by default because neighboring cells are spatially dependent and arbitrary dependence is safer to allow for; return raw p-values and adjusted q-values. Benjamini–Hochberg may be offered as a more powerful alternative only when its independence/positive-dependence condition is justified and declared. Report family size, excluded-cell count, method, and target q level. Do not include insufficient-data cells in the correction or treat their absent p-values as nonsignificant.

FDR controls the expected fraction of false discoveries among rejected hypotheses under its stated assumptions; it does not mean that each mapped cell has a q-level probability of being false. Maps should distinguish raw from FDR-adjusted significance. Spatial dependence can make Benjamini–Hochberg assumptions questionable, hence the conservative default.

## Result labels

Use slope sign for direction and the selected significance threshold for evidence:

- **Increasing trend:** positive slope and significant test (for grids, FDR-adjusted q below the configured level).
- **Decreasing trend:** negative slope and significant test (or adjusted q below level).
- **No statistically significant trend detected:** eligible data, but the test does not pass the configured threshold. Preserve the slope and interval; this is not proof of zero change.
- **Insufficient data:** failed eligibility or unavailable valid test; no direction/significance claim.

Expose statistical significance separately from magnitude. A small but significant slope may have limited practical importance; a large uncertain slope may not be significant. Do not let an LLM recalculate statistics or change these labels.

## Reproducibility and limitations

Each result must include cell coordinates, data-source provenance, parameter and units, input date range, aggregation and coverage rule, valid counts and missingness, test and slope methods, autocorrelation handling, slope interval method/settings, raw p-value, adjusted q-value (if spatial), alpha/q threshold, direction/label, and software/version identifiers. Bootstrap output must include its seed. Keep calculations deterministic for fixed inputs and settings.

These methods detect monotonic temporal patterns. They do not attribute trends to climate change or a particular driver, guarantee homogeneous input data, correct source-product bias, or establish ecological impact. Interpret magnitude alongside data provenance, uncertainty, spatial context, and domain knowledge.

## References

- Hirsch, Slack & Smith (1982), [Techniques of trend analysis for monthly water quality data](https://doi.org/10.1029/WR018i001p00107). Seasonal Kendall for seasonality and missing observations.
- Hirsch & Slack (1984), [A Nonparametric Trend Test for Seasonal Data With Serial Dependence](https://doi.org/10.1029/WR020i006p00727). Dependence-adjusted seasonal rank test and limitations under strong persistence/short records.
- Hamed & Rao (1998), [A modified Mann–Kendall trend test for autocorrelated data](https://doi.org/10.1016/S0022-1694(97)00125-X). Variance correction for autocorrelation.
- Sen (1968), [Estimates of the Regression Coefficient Based on Kendall’s Tau](https://doi.org/10.1080/01621459.1968.10480934). Median pairwise slope and rank-based interval.
- Benjamini & Hochberg (1995), [Controlling the False Discovery Rate](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x); Benjamini & Yekutieli (2001), [The Control of the False Discovery Rate in Multiple Testing under Dependency](https://doi.org/10.1214/aos/1013699998).
