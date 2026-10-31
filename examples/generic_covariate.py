"""Synthetic exposure example: python examples/generic_covariate.py [--filtering MODE]."""
import argparse
import numpy as np
import pandas as pd
from isovita import (
    IsoVitaConfig, PermutationInferenceConfig,
    prepare_isovita_inputs_from_expression, run_isovita_pipeline,
)


def make_inputs():
    rng = np.random.default_rng(17)
    exposure = np.linspace(-3, 3, 60)
    samples = [f"s{i}" for i in range(len(exposure))]
    rows, mapping = [], []
    for group, k in [("mixture_a", 2), ("mixture_b", 4)]:
        latent = (np.tanh(exposure[:, None]) * np.linspace(-0.8, 0.8, k)
                  + rng.normal(0, 0.2, (len(samples), k)))
        fractions = np.exp(latent) / np.exp(latent).sum(axis=1, keepdims=True)
        for j in range(k):
            component = f"{group}_{j}"
            rows.append({"tx_id": component, **dict(zip(samples, 50 * fractions[:, j]))})
            mapping.append({"tx_id": component, "gene_id": group})
    return prepare_isovita_inputs_from_expression(
        expression=pd.DataFrame(rows),
        metadata=pd.DataFrame({"sample_id": samples, "exposure": exposure}),
        tx2gene=pd.DataFrame(mapping), metadata_sample_col="sample_id", time_col="exposure",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtering", choices=["none", "isoplex", "both"], default="none")
    args = parser.parse_args()
    prepared = make_inputs()
    modes = ("none", "isoplex") if args.filtering == "both" else (args.filtering,)
    for mode in modes:
        # IsoPlex is meaningful for transcript TPM. Here the synthetic abundances
        # are on a TPM-like scale solely to demonstrate both API paths.
        config = IsoVitaConfig(
            filtering=mode,
            inference=PermutationInferenceConfig(n_permutations=99, random_seed=17),
        )
        result = run_isovita_pipeline(tpm=prepared.tpm, metadata=prepared.metadata, config=config)
        print(f"\nFiltering: {mode}")
        print(result.balance_summary[["gene_id", "group_A", "group_B"]])
        print(result.continuous_trajectory_results[["gene_id", "f_stat", "p_value", "q_value"]])
    # 99 permutations demonstrate the interface, not production tail resolution.


if __name__ == "__main__":
    main()
