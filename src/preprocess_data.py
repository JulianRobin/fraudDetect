import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the dataset
data_path = "../data/creditcard.csv"
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: The file at {data_path} does not exist.")
    exit()

# Separate features (X) and target (y)
X = data.drop(columns=["Class"])
y = data["Class"]

# Scale 'Amount' and 'Time' columns
scaler = StandardScaler()
X["Time"] = scaler.fit_transform(X["Time"].values.reshape(-1, 1))
X["Amount"] = scaler.fit_transform(X["Amount"].values.reshape(-1, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Save processed data for future use
X_train_resampled.to_csv("../data/X_train_resampled.csv", index=False)
y_train_resampled.to_csv("../data/y_train_resampled.csv", index=False)
X_test.to_csv("../data/X_test.csv", index=False)  # Save X_test
y_test.to_csv("../data/y_test.csv", index=False)  # Save y_test

# Return processed data for further use in model training
X_train_resampled, y_train_resampled, X_test, y_test
