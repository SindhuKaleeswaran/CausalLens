# IHDP benchmark research — Phase 1

Decision date: 2026-11-06. Evidence IDs refer to [references](references.md).

## Original study versus benchmark

IHDP means Infant Health and Development Program. The original eight-site randomized study evaluated early developmental support for premature infants (≤37 weeks) weighing ≤2500 g, aiming to reduce developmental and health problems [R1]. Intervention families received home visits, specialized child-development center education, parent groups and pediatric follow-up; controls received pediatric follow-up/referral. The trial included 377 intervention and 608 control infants [R2].

Hill (2011) introduced the benchmark lineage [R3, R5]. The NPCI implementation removes treated children whose mothers are nonwhite, retaining controls [R4]. This selection disrupts randomization in the retained sample. Child/maternal baseline covariates and treatment labels derive from real study records; benchmark outcomes are generated rather than clinical measurements [R5]. Consequently the benchmark's simulated treatment effect is not the original trial's clinical effect.

## Release decision and versions

Use **CFRNet-lineage IHDP-100**, exactly the two author-hosted NPZ files listed in references. R6 describes 100 realizations as the small repository example and distinguishes the full 1000-realization ICML experiments. The repetition count describes simulated realizations, not independent patient cohorts. R5 reports 747 units (139 treated, 608 control) and 25 features; actual archive dimensions and split counts will be measured in Phase 2. Preserve the supplied train/test partitions; create any later validation subset only from training.

The ICML paper describes NPCI **setting A** [R5]. In the inspected NPCI revision, A uses an exponential control mean and linear treated mean with an offset; B/C use a different polynomial construction [R4]. Labels such as “Hill response surface B” in later literature must not be equated mechanically with NPCI's argument name. Exact archive bytes, source revision and split protocol identify our release. We have not regenerated these archives or verified the exporter revision, simulation seeds, or numerical equality to another library. CSV releases, regenerated outcomes, alternative scaling, different replication counts and library-specific splitting must be checked separately; “IHDP” alone is insufficient provenance.

## Variables and ground truth

For unit i, X contains 25 ordered covariates, T is binary treatment, and Y = yf is the simulated outcome corresponding to assigned T. Only these observed variables enter training. Treatment is a separate model input, not an extra covariate silently appended by the loader.

`mu0` and `mu1` are noiseless expected potential outcomes; `ycf` is the simulated outcome for the unassigned arm [R7]. The evaluation target is `tau_i = mu1_i - mu0_i`. For an explicitly selected split and replication, finite-sample ATE is the mean of tau; ATT averages tau only among T=1. Heterogeneous effects vary across X. The realized contrast reconstructed from yf/ycf contains outcome noise and need not equal tau. In particular, do not assert `yf == T*mu1 + (1-T)*mu0`.

Do not use mu0, mu1, tau or ycf for features, preprocessing fitting, factual target replacement, model selection or tuning. Keep oracle data behind a separate evaluation accessor. Retain original outcome units; no clinical unit is assigned to these simulated numbers.

## Covariates, selection and identification

The upstream source gives an ordered list of 25 names, with six multi-valued numeric fields followed by indicators [R4]; [data dictionary](data_dictionary.md) preserves the distinction between upstream names and NPZ-verified identities. Age, birth characteristics, maternal circumstances and site may be useful pre-intervention descriptors. Association or treatment imbalance does not prove each field is a causal confounder. The excluded ethnicity variable is part of the known selection mechanism and is absent from the selected upstream list.

Selection can leave weak or absent support for treatment comparisons. Full-population ATE estimation can require extrapolation even when an ATT target has better support. A known simulated oracle permits scoring estimates despite difficult observational identification; it does not establish positivity, exchangeability or transportability. Finite overlap diagnostics remain for Phase 3. No estimated propensity model or causal estimator is implemented here.

## Limitations and open evidence

- Semi-synthetic responses test estimators under one generator, not real medical effectiveness.
- Repeated realizations are dependent through underlying covariates; never pool them as independent patients, or match rows across replications by index.
- Upstream normalization may already have used the retained cohort; preserving this published benchmark cannot undo its construction. Any additional learned preprocessing must be train-only.
- No patient identifiers or explicit row mapping are promised by the distributor; exact duplicate-row checks cannot certify patient-level separation.
- Exact exporter lineage, complete per-column coding/units, publisher hashes and standalone redistribution permission remain unresolved. Generic feature identifiers remain canonical.

Phase 1 establishes a source-supported release choice and model/evaluation boundary. Proceed to local Phase 2 acquisition; do not redistribute archives.
