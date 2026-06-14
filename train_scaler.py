import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the original dataset used for training (Replace with actual path)
train_data_path = "D:/b21ci018/project/bvp_features.csv"  # CHANGE THIS
df = pd.read_csv(train_data_path)

# Drop non-feature columns
X_train = df.drop(columns=["Subject", "Segment", "Label"], errors="ignore")

# Fit the scaler properly
scaler = StandardScaler()
scaler.fit(X_train)  # Train the scaler with correct feature names

# Save the new scaler
scaler_path = "D:/b21ci018/project/scaler_fixed.pkl"
joblib.dump(scaler, scaler_path)

print("✅ New scaler saved successfully!")
