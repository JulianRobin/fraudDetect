
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load preprocessed data
X_train = pd.read_csv("../data/X_train_resampled.csv")
y_train = pd.read_csv("../data/y_train_resampled.csv").squeeze()  # Convert to Series
X_test = pd.read_csv("../data/X_test.csv")
y_test = pd.read_csv("../data/y_test.csv").squeeze()  # Convert to Series

# Initialize LightGBM model
lgbm_model = lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 is_unbalance=True,  # Handles class imbalance
                                 random_state=42,
                                 n_estimators=500,
                                 learning_rate=0.1,
                                 max_depth=10,
                                 num_leaves=70)

# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create an ensemble of models using voting classifier
ensemble_model = VotingClassifier(estimators=[('lgbm', lgbm_model), ('rf', rf_model)], voting='soft')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Evaluate the ensemble model
y_pred = ensemble_model.predict(X_test)
y_pred_prob = ensemble_model.predict_proba(X_test)[:, 1]  # Probability estimates for ROC-AUC

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")
