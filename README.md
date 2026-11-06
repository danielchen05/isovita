# IsoVita

**Interpretable compositional features associated with continuous covariates.**

IsoVita learns one interpretable component balance per group, then tests its
association with a continuous coordinate (`time`) using complementary **discrete** (changepoint) and
**continuous** (spline) tests. Both use selection-aware time permutations and
separate Benjamini-Hochberg correction. A group may be detected by either or both.
`time` can represent age, exposure, pseudotime, or another continuous coordinate;
negative values are supported.

## Manuscript

**Title:** *Learning interpretable compositional features associated with continuous covariates*  
**Authors:** Xingyi Chen, Stephanie C. Hicks  
**Venue:** bioRxiv (2026)  
**DOI:** https://doi.org/10.64898/2026.11.04.764349

If you use IsoVita, please cite the manuscript above.

## Installation

Requires **Python 3.10-3.12**.

```bash
pip install "isovita[isoplex]"  # transcript analyses (default filtering)
# Or: pip install isovita    # prefiltered components, filtering="none"
```

Optional extras: `isovita[adata]` for AnnData input and `isovita[notebook]` for
Jupyter. NumPy <2 and pandas <3 bounds are retained for numerical continuity
and compatibility with IsoPlex. From this repository, install `pip install ".[isoplex]"`;
for development use `pip install -e ".[dev,isoplex]"`.

## Quick start

Prepare two tables:

- **TPM:** `gene_id`, unique `transcript_id`, then sample columns containing finite,
  nonnegative transcript TPM (or prefiltered component abundances/proportions).
- **Metadata:** unique `sample_id` and finite numeric `time`, with rows in exactly
  the same order as the TPM sample columns.

Remove missing/empty IDs and nonfinite times before analysis. Use consistent time units.

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

The self-contained [quick-start notebook](https://github.com/danielchen05/isovita/blob/main/examples/quickstart.ipynb)
demonstrates preparation, inference, balance interpretation, plotting, and optional
windows using synthetic data. Its 99 permutations are for demonstration only.
After installing the notebook extra, register a kernel from the same Python environment:

```bash
python -m ipykernel install --user --name isovita --display-name "Python (IsoVita)"
```

For raw expression and transcript-to-gene mapping, use
`prepare_isovita_inputs_from_expression(..., time_col="exposure",
metadata_sample_col="sample_id")`. Expression has `tx_id` plus sample
columns; the mapping has `tx_id` and `gene_id`. The function aligns samples,
annotates groups, and retains negative coordinates, returning `.tpm`,
`.metadata`, and `.summary`. `prepare_isovita_inputs` accepts AnnData
`obsm["abundance"]`; see `isovita-prepare --help` for the preparation CLI.

## Filtering and generic compositions

`IsoVitaConfig(filtering="isoplex")` uses the unchanged TPM eligibility screen
(strictly >0.5 TPM in at least 5 samples by default), positive-total filter, and
IsoPlex effective-isoform selection. Missing IsoPlex raises an installation hint.

`IsoVitaConfig(filtering="none")` bypasses both the TPM screen and IsoPlex.
Supply appropriately prefiltered components; all are marked eligible, with no
abundance cutoff. Values must be finite and nonnegative, and every group's total
must be finite and positive in every sample. Groups with fewer than two components
are recorded as `exclude_lt2` and not tested. Zeros within valid compositions use
the unchanged pseudocount. Minimum sample requirements still apply.

For compatibility, `tpm`, `gene_id`, `transcript_id`, and gene/isoform-oriented helper
names remain: they denote abundance, group ID, and component ID in generic data.
Component IDs must be globally unique. See [generic_covariate.py](examples/generic_covariate.py)
for a synthetic exposure example demonstrating both modes. Prefiltering should
not use the tested coordinate: permutations do not repeat external filtering.

The CLI prepares AnnData inputs only (it does not run inference):
`isovita-prepare --adata data.h5ad --metadata metadata.tsv --tx2gene mapping.csv
--time-col exposure --output-dir prepared`. It requires the `adata` extra.
Apply range restrictions explicitly to metadata before preparation.

## Results

| Attribute | Contents |
| --- | --- |
| `balance_summary`, `balance_loadings` | Selected A/B transcript groups, PC1 loadings, and fit diagnostics. |
| `balance_feature_matrix` | Genes by samples; natural-log ratio `log(GM(A) / GM(B))`. |
| `discrete_trajectory_results` | Changepoints, effects, statistics, and p/q values. |
| `continuous_trajectory_results` | Spline statistics, fit diagnostics, and p/q values. |
| `gene_trajectory_summary`, `final_features` | Combined global evidence and ranked discoveries. |

`p_value` is selection-aware; `analytic_p_value` is diagnostic only. Check `tested`
and balance `status`. Significant subsets are available in the corresponding
`*_trajectory_candidates` tables.

## Optional time windows

```python
from dataclasses import replace
from isovita import WindowConfig

window_result = run_isovita_pipeline(
    tpm=tpm, metadata=metadata,
    branches=("discrete", "continuous", "localized"),
    config=replace(config, windows=WindowConfig(edges=(20, 40, 60, 85))),
)
```

The `localized` flag runs both tests within predefined windows. Balances and
component membership are determined globally; permutations shuffle times across the
full cohort and relearn the balance. BH correction pools gene x window tests
separately for each detector. Intervals are left-closed/right-open, except the
final upper edge is included. Small or degenerate windows are untestable.

Results appear in `localized_discrete_results` and `localized_continuous_results`
and their candidate tables; they do not enter `final_features`.

Filtering provenance is in `filtering_method`, `filtered_expression_df`, and
`filter_summary`. In `none` mode, `global_metrics` is only the membership table
(`gene_id`, `transcript_id`, `effective`); `sample_metrics` is empty. Exports include
`filter_summary.csv`, `component_membership.csv`, and `filtering_method.txt`.
IsoPlex metric CSVs are written only in IsoPlex mode. Use a fresh output directory
for each run to avoid retaining files from earlier runs.
