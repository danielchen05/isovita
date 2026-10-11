import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from isovita import IsoVitaConfig, PermutationInferenceConfig, run_isovita_pipeline

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def inputs():
    return pd.read_csv(FIXTURES / "tpm.csv"), pd.read_csv(FIXTURES / "metadata.csv")


@pytest.fixture(scope="session")
def config():
    return IsoVitaConfig(
        inference=PermutationInferenceConfig(n_permutations=39, random_seed=17, batch_size=7)
    )


@pytest.fixture(scope="session")
def baseline():
    return json.loads((FIXTURES / "baseline.json").read_text())


@pytest.fixture(scope="session")
def result(inputs, config):
    tpm, metadata = inputs
    return run_isovita_pipeline(tpm=tpm, metadata=metadata, config=config)


def assert_frame(actual, expected, *, exact=False):
    assert list(actual.columns) == expected["columns"]
    assert list(actual.index) == expected["index"]
    assert actual.index.name == expected["index_name"]
    assert actual.columns.name == expected["columns_name"]
    for row, exp in zip(actual.to_numpy(), expected["data"]):
        for a, b in zip(row, exp):
            if isinstance(b, list):
                assert list(a) == b
            elif isinstance(b, (float, int)) and not isinstance(b, bool):
                np.testing.assert_allclose(
                    a, b, rtol=0 if exact else 1e-9, atol=0 if exact else 1e-11, equal_nan=True
                )
            else:
                assert a == b
