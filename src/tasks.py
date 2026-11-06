import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sns.set_style('whitegrid')


def load_data(red_path: str, white_path: str) -> pd.DataFrame:
    """
    Loading and merging red and white wine datasets.

    Note: Don't forget to add the 'type' attribute.
    """
    # TODO: Your code here
    data = None

    return data


def remove_outliers(df: pd.DataFrame, features: pd.Index) -> pd.DataFrame:
    """Removing outliers from a dataset using the IQR method."""
    df_clean = df.copy()

    # TODO: Your code here

    return df_clean


def encode_categorical_variables(data: pd.DataFrame) -> pd.DataFrame:
    """Encoding of categorical variables."""
    data_encoded = data.copy()

    # TODO: Your code here

    return data_encoded


def scale_features(
    X_train: pd.DataFrame
) -> np.ndarray:
    """Standartization (scaling) of features"""
    # TODO: Your code here
    X_train_scaled = None

    return X_train_scaled


def apply_pca(X: pd.DataFrame) -> np.ndarray:
    """
    Application of the principal component analysis (PCA) method.
    """
    X_scaled = scale_features(X)

    # TODO: Your code here

    X_pca = None
    return X_pca


def perform_clustering(X_pca: np.ndarray) -> dict[str, float | np.ndarray]:
    """Performing clustering based on PCA results."""
    # K-means
    # TODO: Your code here

    # Model evaluation
    # TODO: Your code here
    results = {'labels': [], 'silhouette_score': 0.0}

    return results


def main():
    # Loading data
    data = load_data('winequality-red.csv', 'winequality-white.csv')

    # Removing outliers
    features = data.columns[:-2]  # Without 'quality' and 'type'
    data = remove_outliers(data, features)

    # Encoding categorical variables
    data = encode_categorical_variables(data)

    # Unsupervised learning task
    X_features = data.drop(['quality'], axis=1)
    X_pca = apply_pca(X_features)
    clustering_results = perform_clustering(X_pca)
    print("\nClustering Results:")
    print(f"Silhouette coefficient: {clustering_results['silhouette_score']:.4f}")

    # Visualization of clusters
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    hue=clustering_results['labels'],
                    palette='viridis')
    plt.title('K-means Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


if __name__ == "__main__":
    main()
