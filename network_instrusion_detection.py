# ============================================
# Cybersecurity Clustering with DBSCAN
# Scenario: Network Intrusion Detection
# Dataset: UNSW-NB15
# ============================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# ------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------
# Option A: use KaggleHub (easy for Kaggle-hosted dataset)
# pip install kagglehub
USE_KAGGLEHUB = False

# Update this if needed
LOCAL_CSV_PATH = "unsw-nb15/UNSW_NB15_training-set.csv"

def load_unsw_nb15(local_csv_path=LOCAL_CSV_PATH, use_kagglehub=USE_KAGGLEHUB):
    """
    Load UNSW-NB15 dataset from local CSV or KaggleHub.
    Returns a pandas DataFrame.
    """
    if os.path.exists(local_csv_path):
        print(f"Loading local file: {local_csv_path}")
        df = pd.read_csv(local_csv_path)
        return df

    if use_kagglehub:
        try:
            import kagglehub
            # Kaggle dataset page: mrwellsdavid/unsw-nb15
            dataset_path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
            print("Downloaded to:", dataset_path)

            # Try common filenames seen in Kaggle mirrors
            candidate_files = [
                "UNSW_NB15_training-set.csv",
                "UNSW_NB15_testing-set.csv",
                "UNSW-NB15_1.csv",
                "UNSW-NB15_2.csv",
                "UNSW-NB15_3.csv",
                "UNSW-NB15_4.csv",
                "UNSWNB15_1.csv",
                "UNSWNB15_2.csv",
                "UNSWNB15_3.csv",
                "UNSWNB15_4.csv"
            ]

            for root, _, files in os.walk(dataset_path):
                for f in files:
                    if f in candidate_files or f.lower().endswith(".csv"):
                        full_path = os.path.join(root, f)
                        print(f"Loading detected CSV: {full_path}")
                        df = pd.read_csv(full_path)
                        return df

            raise FileNotFoundError("No CSV file found in KaggleHub dataset download.")

        except Exception as e:
            raise RuntimeError(
                "Failed to load dataset from KaggleHub. "
                "Please download the dataset manually from Kaggle and place "
                "'UNSW_NB15_training-set.csv' in the working directory.\n"
                f"Original error: {e}"
            )

    raise FileNotFoundError(
        "Dataset file not found. Please either:\n"
        "1) Download 'UNSW_NB15_training-set.csv' manually and place it in the current folder, or\n"
        "2) Set USE_KAGGLEHUB = True and install kagglehub."
    )

df = load_unsw_nb15()

print("Shape:", df.shape)
print(df.head())
print("\nColumns:\n", df.columns.tolist())

# ------------------------------------------------------
# 2. Basic cleaning and sampling
# ------------------------------------------------------
# Keep a manageable sample size for DBSCAN on normal computers.
# DBSCAN can be computationally expensive on very large datasets.
SAMPLE_SIZE = 15000
RANDOM_STATE = 42

if len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"\nSampled dataset shape: {df.shape}")

# Standardize column names
df.columns = [c.strip() for c in df.columns]

# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Known label columns in many UNSW-NB15 CSVs
possible_label_cols = ["label", "attack_cat"]
available_label_cols = [c for c in possible_label_cols if c in df.columns]

# Separate analysis labels for post-hoc evaluation only
analysis_labels = df[available_label_cols].copy() if available_label_cols else pd.DataFrame(index=df.index)

# Remove label columns from features so clustering remains unsupervised
X = df.drop(columns=available_label_cols, errors="ignore")

# Remove obvious ID-like columns if present
drop_cols = [c for c in ["id"] if c in X.columns]
X = X.drop(columns=drop_cols, errors="ignore")

# ------------------------------------------------------
# 3. Identify numeric and categorical columns
# ------------------------------------------------------
numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\nNumeric columns:", len(numeric_cols))
print("Categorical columns:", len(categorical_cols))

# ------------------------------------------------------
# 4. Preprocessing pipeline
# ------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

X_processed = preprocessor.fit_transform(X)

# ------------------------------------------------------
# 5. Dimensionality reduction with PCA
# ------------------------------------------------------
# Convert sparse matrix to dense only if needed and manageable
if hasattr(X_processed, "toarray"):
    X_dense = X_processed.toarray()
else:
    X_dense = X_processed

# Use PCA to reduce dimensionality before DBSCAN
N_COMPONENTS = 10
pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_dense)

print("\nExplained variance ratio (sum):", pca.explained_variance_ratio_.sum())

# ------------------------------------------------------
# 6. Choose DBSCAN parameters
# ------------------------------------------------------
# Heuristic: inspect k-nearest neighbor distances.
# You may need to tune eps depending on your sample.
MIN_SAMPLES = 10

neighbors = NearestNeighbors(n_neighbors=MIN_SAMPLES)
neighbors_fit = neighbors.fit(X_pca)
distances, indices = neighbors_fit.kneighbors(X_pca)

# Sort k-distance for rough inspection
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(k_distances)
plt.title("k-distance graph for choosing DBSCAN eps")
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{MIN_SAMPLES}-NN distance")
plt.grid(True)
plt.savefig("k-distance-graph.png", dpi=300, bbox_inches="tight")
plt.show()

# Set eps manually after inspecting the graph.
# Good starting range often falls around upper bend of k-distance graph.
EPS = 2.0

# ------------------------------------------------------
# 7. Run DBSCAN
# ------------------------------------------------------
dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, n_jobs=-1)
cluster_labels = dbscan.fit_predict(X_pca)

# Add clustering result
results = X.copy()
results["cluster"] = cluster_labels

for col in available_label_cols:
    results[col] = analysis_labels[col]

# ------------------------------------------------------
# 8. Basic analysis
# ------------------------------------------------------
n_noise = np.sum(cluster_labels == -1)
unique_clusters = sorted(set(cluster_labels))
n_clusters = len([c for c in unique_clusters if c != -1])

print("\n===== DBSCAN RESULTS =====")
print("Number of clusters (excluding noise):", n_clusters)
print("Number of noise points:", n_noise)
print("Noise ratio:", round(n_noise / len(results), 4))

cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
print("\nCluster sizes:")
print(cluster_counts)

# Silhouette score only makes sense if we have at least 2 real clusters
non_noise_mask = cluster_labels != -1
non_noise_clusters = pd.Series(cluster_labels[non_noise_mask]).nunique()

if non_noise_clusters >= 2 and non_noise_mask.sum() > 10:
    sil = silhouette_score(X_pca[non_noise_mask], cluster_labels[non_noise_mask])
    print("\nSilhouette score (excluding noise):", round(sil, 4))
else:
    print("\nSilhouette score not available: fewer than 2 non-noise clusters.")

# ------------------------------------------------------
# 9. Post-hoc evaluation using known labels (optional)
# ------------------------------------------------------
# This does not make the method supervised.
# We only use labels to interpret whether discovered clusters align with attacks.
if "label" in results.columns:
    print("\n===== Cluster vs Binary Label =====")
    # label is usually 0=normal, 1=attack
    ct_label = pd.crosstab(results["cluster"], results["label"], normalize="index")
    print(ct_label)

if "attack_cat" in results.columns:
    print("\n===== Cluster vs Attack Category =====")
    ct_attack = pd.crosstab(results["cluster"], results["attack_cat"], normalize="index")
    print(ct_attack)

# ------------------------------------------------------
# 10. 2D visualization
# ------------------------------------------------------
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca_2d.fit_transform(X_dense)

plot_df = pd.DataFrame({
    "PC1": X_2d[:, 0],
    "PC2": X_2d[:, 1],
    "cluster": cluster_labels
})

plt.figure(figsize=(10, 7))
scatter = plt.scatter(plot_df["PC1"], plot_df["PC2"], c=plot_df["cluster"], s=10)
plt.title("DBSCAN Clusters on UNSW-NB15 (2D PCA projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.savefig("dbscan_clusters.png", dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------------------------------
# 11. Save results
# ------------------------------------------------------
results.to_csv("unsw_nb15_dbscan_results.csv", index=False)
print("\nSaved clustering output to: unsw_nb15_dbscan_results.csv")
