# IsoVita

**Interpretable detection of temporal variation in isoform composition.**

**Citation:** TBD.

IsoVita learns one interpretable transcript balance per gene, then tests its
association with age using complementary **discrete** (changepoint) and
**continuous** (spline) tests. Both use selection-aware age permutations and
separate Benjamini–Hochberg correction. A gene may be detected by either or both.

## Installation

Requires **Python 3.10-3.12**.

```bash
pip install isovita
```

Optional extras: `isovita[adata]` for AnnData input and `isovita[notebook]` for
Jupyter. The supported IsoPlex dependency requires NumPy <2 and pandas <3.

## Quick start

Prepare two tables:

- **TPM:** `gene_id`, unique `transcript_id`, then sample columns containing finite,
  nonnegative transcript TPM.
- **Metadata:** unique `sample_id` and finite numeric `age`, with rows in exactly
  the same order as the TPM sample columns.

Remove missing/empty IDs and nonfinite ages before analysis. Use consistent age units.

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
`prepare_isovita_inputs_from_expression`. Expression has `tx_id` plus sample
columns; the mapping has `tx_id` and `gene_id`. The function aligns samples,
annotates genes, and retains postnatal samples by default, returning `.tpm`,
`.metadata`, and `.summary`. `prepare_isovita_inputs` accepts AnnData
`obsm["abundance"]`; see `isovita-prepare --help` for the preparation CLI.

## Results

| Attribute | Contents |
| --- | --- |
| `balance_summary`, `balance_loadings` | Selected A/B transcript groups, PC1 loadings, and fit diagnostics. |
| `balance_feature_matrix` | Genes × samples; natural-log ratio `log(GM(A) / GM(B))`. |
| `discrete_trajectory_results` | Changepoints, effects, statistics, and p/q values. |
| `continuous_trajectory_results` | Spline statistics, fit diagnostics, and p/q values. |
| `gene_trajectory_summary`, `final_features` | Combined global evidence and ranked discoveries. |

`p_value` is selection-aware; `analytic_p_value` is diagnostic only. Check `tested`
and balance `status`. Significant subsets are available in the corresponding
`*_trajectory_candidates` tables.

## Optional age windows

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
IsoPlex membership are learned globally; permutations shuffle ages across the
full cohort and relearn the balance. BH correction pools gene × window tests
separately for each detector. Intervals are left-closed/right-open, except the
final upper edge is included. Small or degenerate windows are untestable.

Results appear in `localized_discrete_results` and `localized_continuous_results`
and their candidate tables; they do not enter `final_features`.