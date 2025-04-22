
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, make_scorer

# Load preprocessed data
X_train = pd.read_csv("../data/X_train_resampled.csv")
y_train = pd.read_csv("../data/y_train_resampled.csv").squeeze()  # Convert to Series
X_test = pd.read_csv("../data/X_test.csv")
y_test = pd.read_csv("../data/y_test.csv").squeeze()  # Convert to Series

# Define the model
model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', is_unbalance=True, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'num_leaves': [31, 50, 70],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [100, 200, 500]
}

# Define scoring metric
scorer = make_scorer(roc_auc_score, needs_proba=True)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=3, verbose=3)

# Run grid search
print("Starting GridSearchCV for hyperparameter tuning...")
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
print(f"Best Parameters: {grid_search.best_params_}")

# Train the model with the best parameters
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Evaluate with ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")
