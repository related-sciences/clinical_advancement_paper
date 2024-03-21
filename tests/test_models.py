import numpy as np
import pandas as pd
import pytest

from pipelines.analyses.publications.clinical_progression_forecasting.public.analysis import (
    get_imputer,
    get_ridge_regression_pipeline,
    make_pandas_pipeline,
)


def get_data(coefs: list[float], n: int = 100):
    coef = np.array(coefs)
    rng = np.random.RandomState(0)
    n_samples = 10000
    n_features = len(coef)
    intercept = 1
    X = pd.DataFrame(
        rng.normal(loc=1, scale=3, size=(n_samples, n_features))
    ).add_prefix("X")
    y = pd.Series(intercept + (X.values * coef).sum(axis=1))
    return X, y, pd.Series(coef, index=X.columns), intercept


@pytest.mark.parametrize("positive", [True, False])
def test_ridge_regression_coefficient_recovery(positive):
    X, y, coefs, intercept = get_data([0, 1, 2] if positive else [-1, 0, 1], n=1_000)
    feature_names = X.columns.tolist()
    pipe = make_pandas_pipeline(
        *get_ridge_regression_pipeline(
            feature_names, positive=positive, add_imputer=False, add_scaler=False
        )
    ).fit(X, y)
    np.testing.assert_allclose(pipe[-1].intercept_, intercept)
    pd.testing.assert_series_equal(pipe[-1].coef_, coefs.astype(float))


@pytest.mark.parametrize(
    "inference_features", [None, ["X2"], ["X1", "X2"], ["X0", "X1", "X2"]]
)
def test_ridge_regression_inference_features(inference_features):
    X, y, _, _ = get_data([-0.2, 0, 0.2])
    feature_names = X.columns.tolist()
    pipe = make_pandas_pipeline(
        *get_ridge_regression_pipeline(
            feature_names, inference_features=inference_features, add_imputer=False
        )
    ).fit(X, y)
    expected = inference_features or feature_names
    actual = pipe[-1].explain(pipe[:-1].transform(X)).columns.tolist()
    assert actual == expected


def test_imputer():
    X = pd.DataFrame(
        {
            "target__static_1": [1, 1, 2, 2, 3, 3],
            "target__static_2": [1, 1, 2, 2, None, None],
            "target__static_3": [1, 1, None, None, None, None],
            "target__clinical__non_static": [1, 2, None, 1, 2, None],
            "target_id": [1, 1, 2, 2, 3, 3],
            "disease_id": [1, 2, 1, 2, 1, 2],
        }
    ).set_index(["target_id", "disease_id"])
    actual = get_imputer(X.columns.tolist()).fit_transform(X).reset_index()
    expected = pd.DataFrame(
        {
            "target_id": [1, 1, 2, 2, 3, 3],
            "disease_id": [1, 2, 1, 2, 1, 2],
            "target__static_1": [1, 1, 2, 2, 3, 3],
            "target__static_2": [1.0, 1.0, 2.0, 2.0, 1.5, 1.5],
            "target__static_3": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "target__clinical__non_static": [1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
        }
    )
    pd.testing.assert_frame_equal(actual, expected)
