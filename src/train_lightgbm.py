
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load preprocessed data
X_train = pd.read_csv("../data/X_train_resampled.csv")
y_train = pd.read_csv("../data/y_train_resampled.csv").squeeze()  # Convert to Series
X_test = pd.read_csv("../data/X_test.csv")
y_test = pd.read_csv("../data/y_test.csv").squeeze()  # Convert to Series

# Initialize LightGBM model
model = lgb.LGBMClassifier(boosting_type='gbdt',
                           objective='binary',
                           is_unbalance=True,  # Automatically adjusts for class imbalance
                           random_state=42,
                           n_estimators=500,
                           learning_rate=0.1,
                           max_depth=10,
                           num_leaves=70)

# Train the model
print("Training LightGBM model with is_unbalance flag...")
model.fit(X_train, y_train)

# Predict probabilities for the test set
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model using confusion matrix and ROC-AUC score
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")
