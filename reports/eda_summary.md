# Phase 3 EDA — IHDP-100

Replication 0, official training split only. All results describe this replication; no test outcomes or oracle fields were used. No model, propensity estimator or preprocessing was fitted.

**Sample:** 672 observations, 123 treated (18.30%) and 549 controls. Missing/nonfinite entries: 0 in X and 0 in Y.

**Outcome association:** treated mean 6.440808, control mean 2.436110; difference **4.004697**. This comparison mixes different covariate distributions and is not a causal effect.

**Balance:** largest absolute SMD is x25 (0.379630); 17 of 25 exceed 0.1. This conventional descriptive reference is not a significance test or proof of confounding.

## Covariate summaries and balance

For numeric columns SMD divides the mean difference by sqrt((s1²+s0²)/2), using sample variances. For binary columns it uses upper-level proportions and sqrt((p1(1−p1)+p0(1−p0))/2). Binary means below are upper-level frequencies; x14 upper level is 2. No values are recoded in stored/model data. Zero denominators yield zero only for identical means, otherwise undefined.

| Feature | Type | Treated mean/proportion | Control mean/proportion | Treated SD | Control SD | SMD |
|---|---|---|---|---|---|---|
| x01 | numeric | 0.1627 | -0.0477 | 0.9455 | 0.9947 | 0.2168 |
| x02 | numeric | 0.1216 | -0.0404 | 0.9117 | 1.0263 | 0.1668 |
| x03 | numeric | -0.0370 | 0.0121 | 0.8686 | 1.0344 | -0.0514 |
| x04 | numeric | -0.2447 | 0.0593 | 0.9105 | 1.0052 | -0.3169 |
| x05 | numeric | -0.1331 | 0.0304 | 1.0474 | 0.9994 | -0.1597 |
| x06 | numeric | 0.1978 | -0.0394 | 0.9348 | 1.0109 | 0.2436 |
| x07 | binary | 0.5041 | 0.5009 | 0.5020 | 0.5005 | 0.0063 |
| x08 | binary | 0.0650 | 0.0984 | 0.2476 | 0.2981 | -0.1219 |
| x09 | binary | 0.6585 | 0.5009 | 0.4761 | 0.5005 | 0.3235 |
| x10 | binary | 0.2764 | 0.3716 | 0.4491 | 0.4837 | -0.2044 |
| x11 | binary | 0.2602 | 0.2823 | 0.4405 | 0.4505 | -0.0499 |
| x12 | binary | 0.2358 | 0.2204 | 0.4262 | 0.4149 | 0.0366 |
| x13 | binary | 0.3821 | 0.3534 | 0.4879 | 0.4785 | 0.0596 |
| x14 | binary | 0.6016 | 0.4262 | 0.4916 | 0.4950 | 0.3565 |
| x15 | binary | 0.1463 | 0.1403 | 0.3549 | 0.3476 | 0.0174 |
| x16 | binary | 0.9431 | 0.9654 | 0.2326 | 0.1830 | -0.1069 |
| x17 | binary | 0.7073 | 0.5792 | 0.4569 | 0.4941 | 0.2698 |
| x18 | binary | 0.9919 | 0.9563 | 0.0902 | 0.2046 | 0.2254 |
| x19 | binary | 0.1463 | 0.1330 | 0.3549 | 0.3399 | 0.0386 |
| x20 | binary | 0.0488 | 0.1475 | 0.2163 | 0.3550 | -0.3366 |
| x21 | binary | 0.1707 | 0.1457 | 0.3778 | 0.3531 | 0.0686 |
| x22 | binary | 0.0407 | 0.0911 | 0.1983 | 0.2880 | -0.2043 |
| x23 | binary | 0.0163 | 0.0893 | 0.1270 | 0.2854 | -0.3310 |
| x24 | binary | 0.0569 | 0.1530 | 0.2326 | 0.3603 | -0.3174 |
| x25 | binary | 0.2846 | 0.1330 | 0.4530 | 0.3399 | 0.3796 |

## Distributions and unusual values

The six leading numeric columns have archived scaled values; large magnitudes are not automatically errors. No observations are removed. The remaining nineteen columns have two levels; x14 uses {1,2}, the others {0,1}. Feature names remain generic because clinical coding is unverified. Histograms share bins between arms and use density so unequal group sizes do not dominate visual comparison.

## Marginal overlap diagnostics

Eight equal-width bins are defined separately for each numeric covariate using training values. Occupied cells with only one arm are sample sparsity warnings, not proof of population positivity failure. Marginal overlap cannot certify joint overlap. Bin counts are descriptive and sensitive to binning.

- x01: 0 occupied one-arm bins; control counts [8, 42, 39, 63, 86, 116, 106, 89], treated counts [1, 5, 10, 6, 22, 27, 23, 29].
- x02: 2 occupied one-arm bins; control counts [4, 7, 65, 91, 179, 153, 47, 3], treated counts [1, 0, 11, 15, 43, 43, 10, 0].
- x03: 0 occupied one-arm bins; control counts [46, 122, 130, 93, 98, 14, 32, 14], treated counts [7, 26, 28, 34, 20, 4, 3, 1].
- x04: 0 occupied one-arm bins; control counts [234, 0, 186, 0, 0, 78, 0, 51], treated counts [74, 0, 29, 0, 0, 14, 0, 6].
- x05: 1 occupied one-arm bins; control counts [1, 4, 7, 31, 97, 213, 167, 29], treated counts [0, 3, 3, 9, 23, 49, 32, 4].
- x06: 0 occupied one-arm bins; control counts [52, 131, 101, 110, 84, 44, 20, 7], treated counts [6, 19, 23, 30, 26, 13, 5, 1].

For binary features, group proportions of zero or one in the table flag absent observed categories within an arm. The known benchmark selection removes treated children with nonwhite mothers (see dataset_research.md); observed imbalance alone does not prove any measured feature is a causal confounder. Known simulated outcomes are not evidence of real medical effectiveness. No independence across replications is assumed.

## Figures and reproduction

Run `python scripts/eda_ihdp.py` or notebook 02 from the installed environment. Six figures are written locally to ignored reports/figures/: treatment counts, numeric distributions, balance, factual outcomes, marginal overlap, and presentation summary. No row-level data are uploaded.

## Boundary and next step

Phase 3 complete. Phase 4 requires explicit permission. Potential future decisions are optional train-only scaling and explicit handling of x14; none are implemented here. No outcome transformations, resplitting, imputation or feature deletion occurred.
