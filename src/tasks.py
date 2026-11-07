import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import os
sns.set_style('whitegrid')

def load_data(red_path: str, white_path: str) -> pd.DataFrame:
    """
    Loading and merging red and white wine datasets.
    """
    def resolve_path(file_path: str) -> str:
        if os.path.exists(file_path):
            return file_path
        
        parent_path = os.path.join('..', os.path.basename(file_path))
        if os.path.exists(parent_path):
            return parent_path
            
        return file_path

    red_file = resolve_path(red_path)
    white_file = resolve_path(white_path)

    try:
        red_df = pd.read_csv(red_file, sep=';')
        if red_df.shape[1] == 1:
            red_df = pd.read_csv(red_file, sep=',')
    except Exception:
        red_df = pd.read_csv(red_file)

    try:
        white_df = pd.read_csv(white_file, sep=';')
        if white_df.shape[1] == 1:
            white_df = pd.read_csv(white_file, sep=',')
    except Exception:
        white_df = pd.read_csv(white_file)

    red_df['type'] = 'red'
    white_df['type'] = 'white'

    data = pd.concat([red_df, white_df], ignore_index=True)
    return data

def remove_outliers(df: pd.DataFrame, features: pd.Index) -> pd.DataFrame:
    """Removing outliers from a dataset using the IQR method."""
    df_clean = df.copy()

    for col in features:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    return df_clean


def encode_categorical_variables(data: pd.DataFrame) -> pd.DataFrame:
    """Encoding of categorical variables using one-hot encoding."""
    data_encoded = data.copy()
    if 'type' in data_encoded.columns:
        data_encoded = pd.get_dummies(data_encoded, columns=['type'], drop_first=True, dtype=int)
    return data_encoded


def scale_features(X_train: pd.DataFrame) -> np.ndarray:
    """Standardization (scaling) of features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return X_train_scaled


def apply_pca(X: pd.DataFrame) -> np.ndarray:
    """
    Application of the principal component analysis (PCA) method.
    """
    X_scaled = scale_features(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca


def find_optimal_clusters(X_pca: np.ndarray, min_k: int = 2, max_k: int = 10) -> int:
    """
    Optional Grid Search to find the optimal number of clusters 
    giving the highest silhouette score.
    """
    best_k = min_k
    best_score = -1.0

    print("--- Searching for Optimal Number of Clusters ---")
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        print(f"k = {k}: Silhouette Score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"Optimal number of clusters: k = {best_k} (Silhouette Score: {best_score:.4f})\n")
    return best_k


def perform_clustering(X_pca: np.ndarray, n_clusters: int = 3) -> dict[str, float | np.ndarray]:
    """Performing clustering based on PCA results using K-Means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    score = silhouette_score(X_pca, labels)
    results = {'labels': labels, 'silhouette_score': score}

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

    # Optional grid search for optimal clusters
    best_k = find_optimal_clusters(X_pca, min_k=2, max_k=10)

    # Clustering with optimal cluster count
    clustering_results = perform_clustering(X_pca, n_clusters=best_k)
    print("Clustering Results:")
    print(f"Silhouette coefficient: {clustering_results['silhouette_score']:.4f}")

    # Visualization of clusters
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    hue=clustering_results['labels'],
                    palette='viridis')
    plt.title(f'K-means Clustering (k={best_k})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


if __name__ == "__main__":
    main()