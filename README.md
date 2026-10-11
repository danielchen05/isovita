# IsoVita

**Interpretable detection of temporal variation in isoform composition.**

IsoVita learns one transcript balance per gene, then tests its association with age
using complementary **discrete** (changepoint) and **continuous** (spline) tests.
Both use selection-aware age permutations and separate Benjamini–Hochberg correction.
A gene may be detected by either or both tests.

## Install

Python **3.10–3.12** is supported; local validation uses Python 3.12.
From the extracted `isovita/` directory:

```bash
python -m pip install .
# Development and tests:
python -m pip install -e ".[dev]"
```

Optional extras: `.[adata]` for AnnData input and `.[notebook]` for Jupyter.
NumPy <2 and pandas <3 are required by the supported IsoPlex dependency.
Version 0.1.0 is prepared for release; this archive does not imply publication on PyPI.

## Inputs and quick start

- **TPM:** pandas table with `gene_id`, unique `transcript_id`, then sample columns
  containing finite, nonnegative transcript TPM.
- **Metadata:** unique `sample_id` and finite numeric `age`, with rows in exactly
  the same order as the TPM sample columns. Ages should use consistent units.
- Clean missing/empty IDs and nonfinite ages before calling the API; some inherited
  validators stringify missing IDs or do not reject infinite ages.

```python
import pandas as pd
from isovita import (
    IsoVitaConfig, PermutationInferenceConfig,
    run_isovita_pipeline, save_isovita_pipeline_outputs,
)

tpm = pd.read_csv("tpm.tsv", sep="\t")
metadata = pd.read_csv("metadata.tsv", sep="\t")
config = IsoVitaConfig(
    inference=PermutationInferenceConfig(n_permutations=5000, random_seed=17)
)
result = run_isovita_pipeline(
    tpm=tpm, metadata=metadata,
    branches=("discrete", "continuous"), config=config, n_jobs=1,
)
print(result.final_features)
save_isovita_pipeline_outputs(result, "isovita_results")
```

The [quick-start notebook](examples/quickstart.ipynb) is self-contained: it generates
synthetic data, prepares inputs, runs both tests, interprets balances, plots features,
and demonstrates optional windows. Its reduced permutation counts are for demonstration.

For raw expression plus transcript-to-gene mapping, use
`prepare_isovita_inputs_from_expression(expression=..., metadata=..., tx2gene=...,
metadata_sample_col="sample_id", age_col="age")`. Expression has `tx_id` plus sample
columns; the mapping has `tx_id` and `gene_id`. This returns `.tpm`, `.metadata`, and
`.summary`; it aligns samples, drops unmapped transcripts and invalid ages, and
restricts to postnatal ages by default. `prepare_isovita_inputs` is the AnnData adapter
for `obsm["abundance"]`; `isovita-prepare --help` exposes its CLI.

## Optional tests within age windows

```python
from dataclasses import replace
from isovita import WindowConfig

window_config = replace(config, windows=WindowConfig(edges=(20, 40, 60, 85)))
window_result = run_isovita_pipeline(
    tpm=tpm, metadata=metadata,
    branches=("discrete", "continuous", "localized"),
    config=window_config,
)
print(window_result.localized_discrete_results)
print(window_result.localized_continuous_results)
```

`localized` is the original API flag for applying the **same discrete and continuous
tests within windows**, not a third detector. Here the windows are `[20,40)`,
`[40,60)`, and `[60,85]`; the last upper edge is included. For one interval, use two
edges, such as `(20, 40)`. Edges must be finite and strictly increasing.

- The balance is learned from the **full input cohort**, then tested within each
  interval. IsoPlex membership is also determined globally.
- Each permutation shuffles ages across the full cohort, relearns the global
  balance, and tests the samples assigned to each interval. This preserves the
  original global age-exchangeability null; it is not a within-window-only shuffle.
- BH correction pools all tested gene × window pairs separately for the windowed
  discrete and windowed continuous families. Windows should be specified before
  examining results; adaptive window selection is not accounted for.
- Windowed discoveries have separate result/candidate tables and **never enter
  `final_features`**, which remains the union of global branch discoveries.
- Use `branches=("localized",)` for windowed tests alone; both windowed detectors
  run, global result tables and `final_features` are empty. Supplying `windows`
  without the flag does not enable windowed analysis.
- Small/degenerate windows are marked untestable with missing p/q values. Default
  requirements are 10 samples per side for discrete tests and 20 per window for
  continuous tests.

## Results and configuration

| Result attribute | Contents |
| --- | --- |
| `balance_summary` | Selected `group_A`, `group_B`, unused isoforms, fit status and diagnostics. |
| `balance_loadings` | PC1 loadings and A/B/unused assignments for each transcript. |
| `balance_feature_matrix` | Genes × samples; feature = `log(GM(A) / GM(B))`, without an ILR scale multiplier. |
| `discrete_trajectory_results` | `max_f_stat`, `best_change_age`, effects, `p_value`, `q_value`, `fdr_significant`. |
| `continuous_trajectory_results` | `f_stat`, fit diagnostics, `p_value`, `q_value`, `fdr_significant`. |
| `gene_trajectory_summary`, `final_features` | Combined global evidence and ranked global discoveries. |
| `localized_discrete_results`, `localized_continuous_results` | Gene × window evidence, including window boundaries and sample counts. |

Significant subsets are available as `discrete_trajectory_candidates`,
`continuous_trajectory_candidates`, `localized_discrete_candidates`, and
`localized_continuous_candidates`. `p_value` is selection-aware; the continuous
`analytic_p_value` is diagnostic only. Inspect `tested` and balance `status`.

Defaults: TPM >0.5 in ≥5 samples; ≥2 effective isoforms; balance-learning spline
df=4, cubic degree, pseudocount `1e-8`, cosine tolerance `0.01`, ≥20 samples;
discrete minimum side size `max(10, ceil(0.15*n))`; continuous cubic spline df=3,
≥20 samples; branchwise FDR 0.05; 1,000 permutations, seed 0, batch size 128.
Configure these through the corresponding nested dataclasses in `IsoVitaConfig`.
`n_jobs=-1` uses all logical CPUs; positive values choose shared-memory gene threads.

Keep inputs, seed, permutation count and dependency versions fixed for reproducibility.
The minimum p-value is `1/(B+1)`; large testing families may need more than 5,000
permutations. Global/windowed branchwise FDR does not give an FDR guarantee for a
union across families. The method assumes exchangeable samples and has no covariate
adjustment. A single balance may miss secondary transcript trajectories.

## Tests, citation and license

```bash
python -m pytest              # default suite
python -m pytest -m ""        # include slow tests
```

Source docstrings describe the preserved APIs. Regression fixtures contain synthetic
reference outputs from the original implementation; original standalone scripts
are not needed or included.

Citation: *IsoVita: Interpretable detection of temporal variation in isoform
composition* (manuscript in preparation; replace with the final citation when available).
MIT license, copyright 2026 Daniel Chen; confirm attribution and licensing before release.
