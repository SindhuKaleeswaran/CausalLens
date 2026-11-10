# Public-demo data decision — November 9, 2026

The owner authorized a public **synthetic-only** deployment. Hosting, processing,
redistributing or publicly displaying row-level IHDP data is not authorized.

## Source review

- [CFRNet upstream README](https://github.com/clinicalml/cfrnet) links the two
  Fredrik Johansson-hosted IHDP archives. Its repository has an MIT software
  license; the README does not establish a separate redistribution grant for
  externally hosted study-derived archives.
- [Author website](https://www.fredjo.com/) did not establish such a grant in the
  material reviewed. Availability for download is not treated as permission to
  redistribute or host participants' covariates.
- [ICPSR IHDP collection](https://www.icpsr.umich.edu/sites/icpsr/view/collections/2260)
  identifies the underlying trial and the study collections.
- [ICPSR general terms](https://www.icpsr.umich.edu/sites/icpsr/about/policies/terms-of-use)
  restrict redistribution without written agreement, with limited exceptions for
  authorized users. These are relevant background, not proof of the exact license
  governing the derived CFRNet archives. The study-specific terms endpoint could
  not be retrieved during this review.

Conclusion: permissions remain **unestablished**. No rights holder has been
contacted, and no permission has been inferred from other repositories' licenses.
Before enabling IHDP hosting, obtain explicit written confirmation covering the
specific derived archives, cloud processing, row-level display and any outputs.

## Aggregate results

The four committed CSVs contain model/replication-level metrics (600 and 800 rows)
and model-level summaries (6 and 8 rows). Their schemas contain no participant
identifiers or row-level covariate/outcome vectors. Existing artifact validation
and scientific findings are preserved. This technical review does not establish
public-display rights for derived results. Therefore the public UI does not read,
display or provide downloads of these files, nor does it display benchmark plots.
Existing repository artifacts are not removed or altered.

## Public implementation

`deploy/streamlit_app.py` executes only `deploy/public_demo.py`. It does not execute
`app.py`, read data directories, consult secrets, or import an acquisition helper.
The previous opt-in cloud download helper has been removed. Legacy environment
variables cannot switch the public app to IHDP. Static file serving is disabled.

All public examples are generated from a fixed-seed NumPy RNG and explicit
educational formulas in `src/causallens/synthetic_demo.py`. There is no sampling,
fitting or calibration to IHDP. Five generic features have no clinical coding.
The six original estimators and metric/policy implementations are reused without
modification. Training receives only synthetic X, T, Y; oracle arrays are used
only for held-out evaluation. All displays and CSV/JSON downloads identify the
simulation as synthetic. IHDP benchmark claims are withheld, not replaced with
simulated claims. The research dashboard remains exactly as in Phase 6.

## Security review scope

Tracked files were checked for raw archives, model binaries, secret filenames,
private-key headers, common GitHub/OpenAI/AWS credential patterns, and executed
notebook output. None were found; all three tracked notebooks have zero output
cells. This is a targeted source review, not a guarantee against every possible
credential format. The public runtime needs no secrets, uploads or external data
requests. Cloud authentication remains the owner's responsibility.
