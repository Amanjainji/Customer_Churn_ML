# 📊Customer Churn Prediction using Machine Learning

This project focuses on building a machine learning model to predict customer churn for a telecom company. The dataset used is the **Telco Customer Churn** dataset from Kaggle, which includes customer demographic information, account details, and services subscribed.

## 🔍 Problem Statement

Customer churn is a critical issue for telecom companies. By predicting churn early, companies can take proactive steps to retain customers and reduce revenue loss. This project aims to build a predictive model that accurately classifies whether a customer is likely to churn.

## 📂 Dataset

* **Source**: [Kaggle - Telco Customer Churn Dataset]
* **Features**: Includes attributes like `gender`, `SeniorCitizen`, `tenure`, `InternetService`, `Contract`, `MonthlyCharges`
* **Target**: `Churn` (Yes/No)

## 🧠 Machine Learning Models Used

The following classification models were trained and evaluated:

1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **XGBoost Classifier**

After training and evaluation, **Random Forest Classifier** yielded the best accuracy and was selected as the final model.

## ✅ Final Model Selection

* **Chosen Model**: Random Forest Classifier
* **Reason**: It achieved the highest accuracy among the models tested.
* **Model Evaluation**: Includes confusion matrix, accuracy score, precision, recall, and F1-score.

## 🧪 Model Evaluation Results

| Metric    | Score (Random Forest) |
| --------- | --------------------- |
| Accuracy  | 0.79                  |
| Precision | 0.78                  |
| Recall    | 0.75                  |
| F1-Score  | 0.77                  |

## 🛠️ Tools & Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib, Seaborn (for visualization)
* Jupyter Notebook

## 📈 Workflow

1. **Data Preprocessing**

   * Handled missing values
   * Converted categorical variables using encoding
   * Normalized numerical features

2. **Model Training & Evaluation**

   * Trained Decision Tree, Random Forest, and XGBoost models
   * Evaluated using train/test split and classification metrics

3. **Final Model Deployment (Optional)**

   * The best model can be exported using `joblib` or `pickle` for deployment.


## 🚀 Future Improvements

* Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
* Use of cross-validation to improve robustness
* Deployment with a Flask or Streamlit web app

## 🙌 Acknowledgments

* Dataset provided by [Kaggle - Telco Customer Churn]

---
