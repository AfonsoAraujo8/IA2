import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset

df = pd.read_csv("ecommerce_data.csv")

# Features and target

X = df.drop("Converted", axis=1)
y = df["Converted"]

# Feature types

categorical_features = ["TrafficSource", "Device"]

numerical_features = ["Age","PreviousPurchases","TimeOnSite","CartValue","ViewedReviews","PagesVisited","AddedToCart"]

# Preprocessing

preprocessor = ColumnTransformer(transformers=[("cat", OneHotEncoder(), categorical_features),("num", StandardScaler(), numerical_features)])

# Create pipeline

model = Pipeline(steps=[("preprocessor", preprocessor),("classifier", LogisticRegression(max_iter=1000))])

# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train model

model.fit(X_train, y_train)

# Predictions

y_pred = model.predict(X_test)

# Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model

joblib.dump(model, "model.pkl")

print("\nModel saved as model.pkl")