from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans # Import KMeans
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

train_ds = read_csv('Datasets/train.csv', sep=';')
test_ds = read_csv('Datasets/test.csv', sep=';')

# Map target variable 'y' to numerical format
train_ds['y'] = train_ds['y'].map({'yes': 1, 'no': 0})

print("\n--- K-Means for Classification ---")

# 1. Prepare data for modeling
X = train_ds.drop('y', axis=1)
y = train_ds['y']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any)
)

# 2. Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Diagnostic Prints ---
print("\n--- Class Distribution in Training Data ---")
print(y_train.value_counts())

# Create a pipeline for preprocessing
preprocess_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform training data
X_train_processed = preprocess_pipeline.fit_transform(X_train)
X_val_processed = preprocess_pipeline.transform(X_val)

# --- K-Means Model ---
# Initialize KMeans with 2 clusters (since we have two classes: yes/no)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

# Fit KMeans on the preprocessed training data
kmeans.fit(X_train_processed)

# Get cluster labels for training data
train_cluster_labels = kmeans.labels_

# Map clusters to actual class labels based on majority vote in training data
cluster_to_class_map = {}
print("\n--- Class Distribution within Clusters (Training Data) ---")
for cluster_id in np.unique(train_cluster_labels):
    cluster_indices = np.where(train_cluster_labels == cluster_id)
    true_labels_in_cluster = y_train.iloc[cluster_indices]
    print(f"Cluster {cluster_id} (size: {len(true_labels_in_cluster)}):\n{true_labels_in_cluster.value_counts()}")

    if not true_labels_in_cluster.empty:
        majority_class = true_labels_in_cluster.mode()[0]
        cluster_to_class_map[cluster_id] = majority_class
    else:
        cluster_to_class_map[cluster_id] = 0 # Default to 0 or handle as an error

print(f"\nCluster to Class Mapping: {cluster_to_class_map}")

# Predict cluster labels for the validation data
val_cluster_labels = kmeans.predict(X_val_processed)

# Map predicted cluster labels to class labels
y_pred_kmeans = np.array([cluster_to_class_map.get(label, 0) for label in val_cluster_labels])

# Evaluate the K-Means model
print(f"\nAccuracy: {accuracy_score(y_val, y_pred_kmeans):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_kmeans))
