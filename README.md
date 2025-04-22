# Fraud Detection System Using Machine Learning

This project demonstrates the development of a fraud detection system using machine learning techniques. The dataset used contains credit card transaction data, with a binary target variable indicating whether a transaction is fraudulent or not. The project utilizes multiple models to address class imbalance and improve classification performance.

## Project Overview

The goal of this project is to build a machine learning model capable of detecting fraudulent credit card transactions. The dataset is highly imbalanced, with fraudulent transactions being much fewer than legitimate ones, which is a typical challenge in fraud detection. The project uses multiple models to address this imbalance and improve classification performance.

We trained three models:
- **LightGBM**: A powerful gradient-boosting algorithm.
- **Logistic Regression**: A simpler, interpretable model.
- **Ensemble Model**: Combines LightGBM and Random Forest to improve prediction accuracy.

We evaluate the models using metrics such as ROC-AUC, precision-recall curves, and confusion matrices.

## Technologies Used

- **Python 3.x**
- **Libraries**:
  - pandas, NumPy
  - scikit-learn (Logistic Regression, Model Evaluation)
  - LightGBM (Gradient Boosting)
  - imbalanced-learn (SMOTE)
  - Matplotlib, Seaborn (Data Visualization)

## Dataset Description

The dataset used is the **Credit Card Fraud Detection** dataset from Kaggle. It contains 284,807 transactions, with 492 instances labeled as fraudulent (Class = 1) and the rest as legitimate (Class = 0). The dataset is highly imbalanced, with fraudulent transactions making up less than 1% of the data.

### Key Features:
- **Time**: Elapsed time in seconds from the first transaction.
- **Amount**: The amount for each transaction.
- **Class**: 1 for fraudulent transactions, 0 for legitimate transactions.

Due to the class imbalance, we applied **SMOTE** to oversample the minority class.

## Model Selection and Justification

- **LightGBM**: We chose LightGBM for its efficiency and performance on imbalanced datasets. It is a gradient-boosting model that handles large datasets and categorical features well. We used it as our primary model.
  
- **Logistic Regression**: This model was chosen for comparison because it is simple, interpretable, and commonly used for binary classification tasks like fraud detection. We applied class weighting to account for the imbalance.

- **Ensemble Model**: We combined the predictions of LightGBM and Random Forest using a **Voting Classifier** to leverage the strengths of both models. This ensemble method helps improve accuracy by combining diverse model predictions.

## Training and Evaluation Process

### Preprocessing:
- The features **Time** and **Amount** were scaled using `StandardScaler`.
- We used **SMOTE** to oversample the minority class in the training data.

### Model Training:
- **LightGBM**: Trained using default hyperparameters initially and then fine-tuned with GridSearchCV.
- **Logistic Regression**: Trained with class weights to address class imbalance.
- **Ensemble**: Combined the predictions of LightGBM and Random Forest using a voting classifier.

### Evaluation Metrics:
- **ROC-AUC Score**: Measures the model's ability to distinguish between classes.
- **Precision-Recall Curve**: Useful for imbalanced datasets where false positives and false negatives are costly.
- **Confusion Matrix**: Provides insight into false positives and false negatives.

The **Ensemble Model** outperformed the individual models, achieving a ROC-AUC score of 0.99, compared to 0.98 for LightGBM and 0.85 for Logistic Regression.

### Key Evaluation Results:
- **LightGBM**: ROC-AUC = 0.98
- **Logistic Regression**: ROC-AUC = 0.85
- **Ensemble Model**: ROC-AUC = 0.99

## Future Work / Next Steps

- **Model Improvements**: Experiment with XGBoost, Neural Networks, or other ensemble techniques to further improve accuracy.
- **Hyperparameter Optimization**: Perform more extensive hyperparameter tuning using RandomizedSearchCV or Bayesian Optimization.

## License

This project is open source and available under the MIT License.
