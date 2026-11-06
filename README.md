# Practical Labs 6
# Clustering and Dimensionality Reduction

## Description

In this task, you will learn to apply unsupervised machine learning methods to the [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) dataset. This dataset consists of physicochemical properties of red and white wines, as well as quality ratings assigned by experts.

### Data Structure

| Characteristic                  | Variable Name              | Data Type      |
|---------------------------------|----------------------------|----------------|
| Fixed acidity                   | `fixed acidity`            | float          |
| Volatile acidity                | `volatile acidity`         | float          |
| Citric acid                     | `citric acid`              | float          |
| Residual sugar                  | `residual sugar`           | float          |
| Chlorides                       | `chlorides`                | float          |
| Free sulfur dioxide             | `free sulfur dioxide`      | float          |
| Total sulfur dioxide            | `total sulfur dioxide`     | float          |
| Density                         | `density`                  | float          |
| pH level                        | `pH`                       | float          |
| Sulphates                       | `sulphates`                | float          |
| Alcohol content                 | `alcohol`                  | float          |
| __Quality rating__              | `quality`                  | int            |
| __Type__                        | `type`                     | Categorical    |

- Fixed acidity: Concentration of non-volatile acids (e.g., tartaric acid) in g/dm³.
- Volatile acidity: Concentration of volatile acids (e.g., acetic acid) in g/dm³.
- Citric acid: Concentration of citric acid in g/dm³.
- Residual sugar: Amount of sugar remaining after fermentation in g/dm³.
- Chlorides: Concentration of chlorides (salt) in g/dm³.
- Free sulfur dioxide: Amount of free SO₂ in mg/dm³.
- Total sulfur dioxide: Total amount of SO₂ in mg/dm³.
- Density: Density of wine in g/cm³.
- pH level: Measure of acidity or alkalinity.
- Sulphates: Concentration of potassium sulphate in g/dm³.
- Alcohol content: Percentage of alcohol by volume (%).
- Quality rating (__target variable__): Wine quality rating (from 0 to 10) provided by expert wine tasters.
- Type: Indicates the type of wine (`'red'` or `'white'`). The type is not present in the data itself but is part of the filename.

## Preparation

Before completing the task, you need to set up the development environment.

### Creating a Python Virtual Environment

1. For Windows:
   ```powershell
   python -m venv venv
   ```

2. For Linux/macOS:
   ```bash
   python3 -m venv venv
   ```

### Activating the Virtual Environment

1. For Windows:
   ```powershell
   .\venv\Scripts\activate
   ```

2. For Linux/macOS:
   ```bash
   source venv/bin/activate
   ```

> After activating the virtual environment, you will see the prefix `(venv)` before the command line.

### Installing Required Packages from `requirements.txt`

1. For both systems:
   ```bash
   pip install -r requirements.txt
   ```

## Tasks

### Task 1: Data Preprocessing

- Data Loading:
   - Implement the `load_data(red_path, white_path)` function to load red and white wine datasets and merge them into a single DataFrame.
      - The `type` characteristic (`'red'` or `'white'`) must be added to the dataset based on the filename from which the data is read.
      - You'll need to use pandas to read the CSV files and concatenate them.
      - Make sure to handle any potential missing values or inconsistencies between the two datasets.
- Data Preprocessing:
   - Implement the `remove_outliers(df, features)` function to detect and handle outliers in the dataset using the [IQR](https://medium.com/@pp1222001/outlier-detection-and-removal-using-the-iqr-method-6fab2954315d) method.
      - For each feature, calculate Q1 (25th percentile), Q3 (75th percentile), and IQR (Q3 - Q1).
      - Remove data points where the feature value is less than Q1 - 1.5*IQR or greater than Q3 + 1.5*IQR.
      - Return the cleaned DataFrame with outliers removed.
   - Implement the `scale_features(X_train, X_test)` function to standardize features using the Standard Scaler.
      - Use sklearn's StandardScaler to transform the features.
      - The function should return the scaled features as a numpy array.
   - If there are categorical variables in the dataset (e.g., `type`), encode them by implementing the `encode_categorical_variables(data)` function.
      - Convert the 'type' column to numerical values using one-hot encoding.
      - Return the DataFrame with encoded categorical variables.

### Task 2: Unsupervised Learning

- Dimensionality Reduction:
   - Implement the `apply_pca(X)` function, applying Principal Component Analysis (PCA) to reduce the data dimensionality to `2` components for further analysis.
      - Use sklearn's PCA to transform the data.
      - The function should return the transformed data as a numpy array with 2 columns.
- Clustering:
   - Implement the `perform_clustering(X_pca)` function to perform clustering based on PCA results using the K-means algorithm.
      - Use sklearn's KMeans to cluster the data.
      - Choose an appropriate number of clusters (e.g., 3 or 4).
      - Calculate the silhouette score to evaluate the clustering quality.
      - Return a dictionary with 'labels' (cluster assignments) and 'silhouette_score'.
- Determining the Optimal Number of Clusters:
   - _Optional_: Use Grid Search to find the best parameters for your model.
      - Try different numbers of clusters (e.g., from 2 to 10).
      - For each number of clusters, calculate the silhouette score.
      - Identify the number of clusters that gives the highest silhouette score.
