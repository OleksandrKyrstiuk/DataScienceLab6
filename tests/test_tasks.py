import pytest
import pandas as pd
import numpy as np

from tasks import (
    load_data, remove_outliers, scale_features, encode_categorical_variables,
    apply_pca, perform_clustering
)

@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return load_data('winequality-red.csv', 'winequality-white.csv')


def test_load_data(data):
    assert not data.empty, "The data must not be empty."
    assert 'type' in data.columns, "The data must contain a 'type' column."
    assert data['type'].nunique() == 2, "There should be two types of wine."


def test_remove_outliers(data):
    features = data.drop(['quality', 'type'], axis=1)
    data_no_outliers = remove_outliers(data, features)
    assert data_no_outliers.shape[0] < data.shape[0], "The size of the data should decrease after removing outliers."


def test_scale_features(data):
    X = data.drop(['quality', 'type'], axis=1)
    X_train_scaled = scale_features(X)
    means = np.mean(X_train_scaled, axis=0)
    stds = np.std(X_train_scaled, axis=0)
    assert np.allclose(means, 0, atol=1e-7), "The mean of the scaled features of the training set is not equal to 0."
    assert np.allclose(stds, 1, atol=1e-7), "The standard deviation of the scaled features of the training set is not equal to 1."


def test_perform_clustering(data):
    data = encode_categorical_variables(data)
    X = data.drop(['quality'], axis=1)
    X_pca = apply_pca(X)
    results = perform_clustering(X_pca)
    sil_score = results['silhouette_score']
    assert sil_score > 0.4, f"The KMeans silhouette score must be greater than 0.3, but {sil_score:.4f} was obtained."