
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

# Load preprocessed data
X_train = pd.read_csv("../data/X_train_resampled.csv")
y_train = pd.read_csv("../data/y_train_resampled.csv").squeeze()  # Convert to Series
X_test = pd.read_csv("../data/X_test.csv")
y_test = pd.read_csv("../data/y_test.csv").squeeze()  # Convert to Series

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Get predicted probabilities
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluate across different thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
for threshold in thresholds:
    y_pred = (y_pred_prob >= threshold).astype(int)
    print(f"\nThreshold: {threshold}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Plot Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")
