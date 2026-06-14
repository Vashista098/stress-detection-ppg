import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib

# Load extracted features
data = pd.read_csv("D:/b21ci018/project/bvp_features.csv")

# Separate features and labels
X = data.drop(columns=["Subject", "Segment", "Label"])  # Drop non-feature columns
y = data["Label"]  # Target labels

# Train-test split (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5,weights="distance"),
    "Logistic Regression": LogisticRegression()
}

# Train & evaluate models
best_model = None
best_auc = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
    
    print(f"{name} Results:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}\n")
    
    if auc_roc > best_auc:
        best_auc = auc_roc
        best_model = model

# Save best model
joblib.dump(best_model, "D:/b21ci018/project/best_stress_model.pkl")
print("✅ Model training complete. Best model saved as 'best_stress_model.pkl'.")
import joblib



# Save the scaler
scaler = StandardScaler()
scaler.fit(X_train)  # Fit the scaler on training data
joblib.dump(scaler, "D:/b21ci018/project/scaler.pkl")

print("✅ Model and scaler saved successfully!")
