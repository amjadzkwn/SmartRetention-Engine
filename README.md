# 📊 SmartRetention Engine: Telco Churn Predictor

SmartRetention Engine is an end-to-end Machine Learning project designed to predict customer churn for a telecommunications company. By identifying high-risk customers, businesses can implement proactive retention strategies to reduce revenue loss.

Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## 🚀 Project Overview

This project utilizes the Telco Customer Churn dataset to build a predictive model. The final deployment is a Streamlit web application that allows users to input customer data and receive real-time risk assessments.

Key Features:
- Data Cleaning & EDA: Handling missing values and visualizing churn drivers (Tenure, Contract type).
- Imbalanced Data Handling: Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the churn classes.
- Model Optimization: Compared Logistic Regression and Random Forest against an Optimized XGBoost model using GridSearchCV.
- Interactive Dashboard: A user-friendly Streamlit UI for real-time predictions.

## 🏗️ Project Architecture

The project is structured into modular Python scripts for maintainability:

1. data_cleaning_eda.py: Performs data preprocessing and generates statistical visualizations.
2. featureEngineering.py: Handles Label Encoding, One-Hot Encoding, Feature Scaling, and SMOTE.
3. modeling.py: Trains multiple classifiers and exports the best-performing model (XGBoost).
4. streamlit_app.py: The frontend application script.

## 🛠️ Technical Stack

- Language: Python
- Libraries: Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-Learn
- Visualization: Matplotlib, Seaborn
- Deployment: Streamlit
- Model Storage: Joblib

## 📈 Exploratory Data Analysis (EDA)

**Churn Distribution**

![churn_distribution](https://github.com/amjadzkwn/SmartRetention-Engine/blob/4a73a142223a771b7ae217a68394cce474fad62c/eda/1_agihan_churn.png)

**Tenure vs Churn**

![tenure_vs_churn](https://github.com/amjadzkwn/SmartRetention-Engine/blob/571dad36e20d33647d0d31e40cadcdcfb024f5fb/eda/2_tenure_vs_churn.png)

**Contract vs Churn**

![contract_vs_churn](https://github.com/amjadzkwn/SmartRetention-Engine/blob/9143953ecb973866db358d3e6de3554512513469/eda/3_kontrak_vs_churn.png)

**Correlation Heatmap**

![correlation_heatmap](https://github.com/amjadzkwn/SmartRetention-Engine/blob/e77e5942c9bb573e09a8e1890764ad9c7fb4cce1/eda/4_heatmap_korelasi.png)

## 📈 Model Performance

To ensure the model effectively captures customers likely to leave, the training focus was placed on Recall.

**Logistic Regression**

![logistic_regression_cm](https://github.com/amjadzkwn/SmartRetention-Engine/blob/b176de7befa02fe2349bb409fbd0cb1da1d2dca6/confusion_matrix/cm_logistic_regression.png)

**Random Forest**

![random_forest_cm](https://github.com/amjadzkwn/SmartRetention-Engine/blob/69ebc45d7ef824661271a2d8b5a67b28499f8c8f/confusion_matrix/cm_random_forest.png)

**XGBoost**

![xgboost_cm](https://github.com/amjadzkwn/SmartRetention-Engine/blob/2fd4142c243e87755662f7b83fee7fa1b49adf11/confusion_matrix/cm_xgboost_optimized.png)

**Performance Comparison**
|Model|Accuracy|Recall (Churn)|F1-Score|
|-----|--------|--------------|--------|
|Logistic Regression|76.30%|74.60%|0.6256|
|Random Forest|76.72%|63.64%|0.5920|
|XGBoost|73.03%|80.21%|0.6122|

**Why XGBoost?**

While Random Forest achieved a slightly higher accuracy, the Optimized XGBoost model was selected as the final engine because it achieved the highest Recall (80.21%). In a churn prevention context, it is more valuable to correctly identify as many potential churners as possible, even if it results in a few more false alarms.

## 🧪 Testing Scenarios

To ensure the model makes logical business decisions, it was tested against four distinct customer personas. The results demonstrate that the model successfully captures key churn drivers like contract type, tenure, and payment methods.

**1. The High-Risk Newcomer**

Profile: New customer, Fiber optic, expensive monthly charges, no contract.

**Input:**
- **Tenure:** 2 bulan
- **Contract:** Month-to-month
- **Internet Service:** Fiber optic
- **Monthly Charges:** $105.00
- **Online Security / Tech Support:** No
- **Payment Method:** Electronic check
- **Paperless Billing:** Yes

**Output:**
![High-Risk_Newcomer](https://github.com/amjadzkwn/SmartRetention-Engine/blob/c45d281b086490699ce125c0ba0363abc003b560/output_streamlit/high-risk_newcomer.png)

Prediction: **⚠️ CHURN (69.16%)**

Insight: The model correctly identifies that high-cost services combined with a lack of contractual commitment in the first few months represent a major risk.

**2. The Loyal Veteran**

Profile: Long-term customer (70 months), Two-year contract, low monthly charges.

**Input:**
- **Tenure:** 70 bulan
- **Contract:** Two year
- **Internet Service:** DSL
- **Monthly Charges:** $40.00
- **Online Security / Tech Support:** Yes
- **Payment Method:** Credit card (automatic)
- **Dependents / Partner:** Yes

**Output:**
![Loyal_Veteran](https://github.com/amjadzkwn/SmartRetention-Engine/blob/e540524d18d2cd35361bb5b3810e67215802b806/output_streamlit/loyal_veteran.png)

Prediction: **✅ NOT CHURN (22.11%)**

Insight: High tenure and long-term contracts act as strong "anchors," significantly lowering the churn probability.

**3. The Vulnerable Senior**

Profile: Senior citizen, One-year contract, uses Tech Support, manual payment.

**Input:**
- **Senior Citizen:** Yes
- **Tenure:** 18 bulan
- **Contract:** One year
- **Internet Service:** Fiber optic
- **Tech Support:** Yes
- **Online Security:** No
- **Payment Method:** Mailed check
- **Monthly Charges:** $85.00

**Output:**
![Vulnerable_Senior](https://github.com/amjadzkwn/SmartRetention-Engine/blob/27492bbd80fb66064703744a25ddebed86afd47c/output_streamlit/vulnerable_senior.png)

Prediction: **✅ NOT CHURN (29.40%)**

Insight: Despite being a senior citizen (often a higher risk group), the presence of a contract and use of support services successfully pulls the risk down into the "safe" zone.

**4. The Unhappy Power User**

Profile: 2-year tenure, but still on a Month-to-month contract with high charges.

**Input:**
- **Tenure:** 24 bulan
- **Contract:** Month-to-month
- **Internet Service:** Fiber optic
- **Streaming TV / Movies:** Yes
- **Multiple Lines:** Yes
- **Monthly Charges:** $110.00
- **Online Security:** No

**Output:**
![Unhappy_Power_User](https://github.com/amjadzkwn/SmartRetention-Engine/blob/4a73a142223a771b7ae217a68394cce474fad62c/eda/1_agihan_churn.png)

Prediction: **⚠️ CHURN (63.61%)**

Insight: This is a crucial "Red Flag" scenario. The model recognizes that even with a decent tenure, a customer is highly likely to leave if they are paying high prices without the stability of a fixed contract.
