from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

train_ds = read_csv('Datasets/train.csv', sep=';')
test_ds = read_csv('Datasets/test.csv', sep=';')

# Map target variable 'y' to numerical format
train_ds['y'] = train_ds['y'].map({'yes': 1, 'no': 0})

print("\n--- Model Training ---")

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

# --- Linear Regression Model ---
print("\n--- Linear Regression ---")

# Create the pipeline with preprocessing and the model
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', LinearRegression())])

# Train
lr_pipeline.fit(X_train, y_train)

# Predict
y_pred_lr = lr_pipeline.predict(X_val)

# Evaluate
mse = mean_squared_error(y_val, y_pred_lr)
r2 = r2_score(y_val, y_pred_lr)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# For classification, we can threshold the output
y_pred_lr_class = [1 if pred > 0.5 else 0 for pred in y_pred_lr]
print(f"Accuracy (threshold at 0.5): {accuracy_score(y_val, y_pred_lr_class):.4f}")


# --- Logistic Regression Model (Better for this problem) ---
print("\n--- Logistic Regression (Recommended) ---")

# Create the pipeline
log_reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', LogisticRegression(random_state=42, max_iter=1000))])

# Train the model
log_reg_pipeline.fit(X_train, y_train)

# Make predictions
y_pred_log_reg = log_reg_pipeline.predict(X_val)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_val, y_pred_log_reg):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_log_reg))