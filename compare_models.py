import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset

df = pd.read_csv("ecommerce_data.csv")

# Features and target

X = df.drop("Converted", axis=1)
y = df["Converted"]

# Feature groups

categorical_features = [
    "TrafficSource",
    "Device"
]

numerical_features = [
    "Age",
    "PreviousPurchases",
    "TimeOnSite",
    "CartValue",
    "ViewedReviews",
    "PagesVisited",
    "AddedToCart"
]

# Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ]
)

# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
}

# Train and evaluate models

results = []

for model_name, classifier in models.items():

    print("\n" + "=" * 50)
    print(f"{model_name}")
    print("=" * 50)

    # Create pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy
    })

    # Print results
    print(f"\nAccuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Final comparison table

results_df = pd.DataFrame(results)

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

print(results_df)

# FEATURE IMPORTANCE

print("\n" + "=" * 50)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 50)

# Recreate Random Forest pipeline
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ))
])

# Train
rf_pipeline.fit(X_train, y_train)

# Get transformed feature names
feature_names = rf_pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

# Get importances
importances = rf_pipeline.named_steps[
    "classifier"
].feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort descending
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df)