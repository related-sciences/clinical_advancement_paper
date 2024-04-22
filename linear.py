from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted


def create_constraints(
    all_features: list[str],
    *,
    positive_features: list[str] | None,
    negative_features: list[str] | None,
) -> list[dict[str, Any]]:
    constraints = []

    if not isinstance(all_features, list):
        raise ValueError("all_features must be a list")

    if positive_features is not None:
        if not isinstance(positive_features, list):
            raise ValueError("positive_features must be a list")
        if invalid_features := set(positive_features) - set(all_features):
            raise ValueError(
                f"The following positive_features are not present in X: {invalid_features}"
            )
        for feature in positive_features:
            feature_index = all_features.index(feature)
            constraints.append(
                {"type": "ineq", "fun": lambda params, idx=feature_index: params[idx]}
            )

    if negative_features is not None:
        if not isinstance(negative_features, list):
            raise ValueError("negative_features must be a list")
        if invalid_features := set(negative_features) - set(all_features):
            raise ValueError(
                f"The following negative_features are not present in X: {invalid_features}"
            )
        for feature in negative_features:
            feature_index = all_features.index(feature)
            constraints.append(
                {"type": "ineq", "fun": lambda params, idx=feature_index: -params[idx]}
            )

    return constraints


def check_binary_target(y: pd.Series) -> pd.Series:
    if not (target_type := type_of_target(y)) == "binary":
        raise ValueError(f"y must be binary; got {target_type}")
    if not is_numeric_dtype(y):
        raise ValueError(f"y must be numeric; got {y.dtype}")
    y = y.astype(int)
    if not y.isin([0, 1]).all():
        raise ValueError(
            f"y must be binary value in {{0, 1}}; got values: {y.unique()}"
        )
    return y


class ConstrainedEstimator(BaseEstimator, ABC):  # type: ignore[misc]
    def __init__(
        self,
        inference_features: list[str] | None = None,
        positive_features: list[str] | None = None,
        negative_features: list[str] | None = None,
        C: float | None = 1.0,
        max_iter: int | None = None,
    ):
        self.inference_features = inference_features
        self.positive_features = positive_features
        self.negative_features = negative_features
        self.C = C
        self.max_iter = max_iter

    @abstractmethod
    def _objective(self, params: Any, X: Any, y: Any) -> Any:
        pass

    @abstractmethod
    def _jacobian(self, params: Any, X: Any, y: Any) -> Any:
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConstrainedEstimator":
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series")
        self._check_feature_names(X, reset=True)
        all_features = list(X.columns)
        constraints = create_constraints(
            all_features,
            positive_features=self.positive_features,
            negative_features=self.negative_features,
        )
        n_features = X.shape[1]
        initial_params = np.zeros(n_features + 1)
        result = minimize(
            **dict(  # noqa: C408
                fun=self._objective,
                jac=self._jacobian,
                x0=initial_params,
                args=(X.values, y.values),
                method="SLSQP",
                constraints=constraints,
            ),
            **(
                {}
                if self.max_iter is None
                else dict(options=dict(maxiter=self.max_iter))  # noqa: C408
            ),
        )
        self.optimizer_success_ = result.success
        self.optimizer_message_ = result.message
        self.optimizer_code_ = result.status
        self.optimizer_niter_ = result.nit
        self.coef_ = pd.Series(result.x[:-1], index=all_features)
        if self.inference_features:
            self.coef_ = self.coef_.loc[self.inference_features]
        self.intercept_ = result.x[-1]
        assert self.coef_.ndim == 1
        assert self.intercept_.ndim == 0
        return self

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        check_is_fitted(self)
        if self.inference_features is not None:
            X = X[self.inference_features]
        assert X.columns.to_list() == self.coef_.index.to_list()
        E = X.mul(self.coef_.values, axis="columns")
        assert E.shape == X.shape
        assert isinstance(E, pd.DataFrame)
        return E


EPS = 1e-16


class ClassifierObjectiveMixin:
    def _objective(self: ConstrainedEstimator, params: Any, X: Any, y: Any) -> Any:  # type: ignore[misc]
        coef, intercept = params[:-1], params[-1]
        z = X.dot(coef) + intercept
        p = np.clip(expit(z), EPS, 1 - EPS)
        log_likelihood = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        regularization = 0.0 if self.C is None else (0.5 * np.sum(coef**2) / self.C)
        return -log_likelihood + regularization

    def _jacobian(self: ConstrainedEstimator, params: Any, X: Any, y: Any) -> Any:  # type: ignore[misc]
        coef, intercept = params[:-1], params[-1]
        z = X.dot(coef) + intercept
        p = np.clip(expit(z), EPS, 1 - EPS)
        grad_coef = X.T.dot(p - y)
        if self.C is not None:
            grad_coef += coef / self.C
        grad_intercept = np.sum(p - y)
        assert grad_coef.shape == coef.shape
        assert grad_intercept.ndim == 0
        return np.concatenate((grad_coef, [grad_intercept]))


class ClassifierPredictionMixin(ClassifierMixin):  # type: ignore[misc]
    def predict_proba(self, X: pd.DataFrame) -> Any:
        p = expit(self.intercept_ + self.explain(X).sum(axis="columns"))
        return np.column_stack((1 - p, p))

    def predict(self, X: pd.DataFrame) -> Any:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class ConstrainedLogisticClassifier(
    ClassifierObjectiveMixin, ClassifierPredictionMixin, ConstrainedEstimator
):
    """
    Constrained Logistic Regression Classifier.

    This class implements a logistic regression classifier with the ability to impose constraints
    on the model coefficients. These can be specified as either positive or negative constraints
    on specific features. This implementation also supports the use of feature subsets for inference.

    Parameters
    ----------
    inference_features : list of str, optional (default=None)
        The list of feature names to be used for inference. If None, all features will be used.
        All features are always used for training, this only controls features used to make predictions.
    positive_features : list of str, optional (default=None)
        The list of feature names that should have positive coefficients.
    negative_features : list of str, optional (default=None)
        The list of feature names that should have negative coefficients.
    C : float, optional (default=1.0)
        The regularization strength. Smaller values specify stronger regularization.
    max_iter : int, optional (default=None)
        The maximum number of iterations for the optimizer.

    Attributes
    ----------
    coef_ : array of shape (n_features,)
        The estimated coefficients of the model.
    intercept_ : float
        The estimated intercept of the model.
    optimizer_success_ : bool
        Whether the optimizer successfully converged.
    optimizer_message_ : str
        The message returned by the optimizer.
    optimizer_code_ : int
        The status code returned by the optimizer.
    optimizer_niter_ : int
        The number of iterations performed by the optimizer.

    Examples
    --------
    >>> import pandas as pd
    >>> X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    >>> y = pd.Series([0, 1, 1])
    >>> clf = ConstrainedLogisticClassifier(positive_features=['feature1'])
    >>> clf.fit(X, y)
    ConstrainedLogisticClassifier(...)
    >>> clf.predict(X)
    array([0, 1, 1])
    >>> clf.predict_proba(X)
    array([[0.73105858, 0.26894142],
           [0.26894142, 0.73105858],
           [0.11920292, 0.88079708]])
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConstrainedLogisticClassifier":
        y = check_binary_target(y)
        return super().fit(X, y)
