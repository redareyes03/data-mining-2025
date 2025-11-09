from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

train_ds = read_csv('Datasets/train.csv', sep=';')
test_ds = read_csv('Datasets/test.csv', sep=';')

train_ds['y'] = train_ds['y'].map({'yes': 1, 'no': 0})

print("\n--- K-Nearest Neighbors Classification ---")

X = train_ds.drop('y', axis=1)
y = train_ds['y']

categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any)
)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- K-Nearest Neighbors Model ---
print("\n--- K-Nearest Neighbors ---")

# pipeline for KNN
knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', KNeighborsClassifier(n_neighbors=5))]) # Using 5 neighbors as a default

# Train the KNN model
knn_pipeline.fit(X_train, y_train)

# Make predictions with KNN
y_pred_knn = knn_pipeline.predict(X_val)

# Evaluate the KNN model
print(f"Accuracy: {accuracy_score(y_val, y_pred_knn):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_knn))
