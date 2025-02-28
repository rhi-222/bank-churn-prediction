# Bank Churn Prediction

## Table of Contents
- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Modeling Approach](#modeling-approach)
- [Results & Model Performance](#results--model-performance)
- [Key Insights & Business Impact](#key-insights--business-impact)
- [Next Steps & Potential Improvements](#next-steps--potential-improvements)
- [How to Use](#how-to-use)
- [Connect With Me](#connect-with-me)

---

## Overview
This project aims to predict **customer churn** for Beta Bank using **supervised machine learning models**. The goal is to identify customers at risk of leaving and enable targeted retention strategies.

## Business Problem
Beta Bank is experiencing customer churn, where clients are terminating their accounts. Since **retaining customers is more cost-effective than acquiring new ones**, the bank wants to:
- Predict **which customers are likely to leave**.
- Identify **key factors influencing churn** (e.g., age, credit score, account balance).
- Develop **data-driven retention strategies**.

## Dataset
The dataset consists of customer records, including demographics, banking activity, and whether they churned. The data is stored in the **`data/` folder**.

### **Dataset File:**
- **`data/Churn.csv`** → Customer banking records.

### **Key Features:**
- **Demographics:** `Geography`, `Gender`, `Age`
- **Banking Activity:** `CreditScore`, `Balance`, `NumOfProducts`
- **Customer Behavior:** `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
- **Target Variable:** `Exited` (1 = Churned, 0 = Stayed)

---

## Exploratory Data Analysis (EDA)
### **Key Findings:**
- **Churn rate:** A significant portion of customers left the bank.
- **Age impact:** Older customers were more likely to churn.
- **Credit Score influence:** Higher credit scores correlated with lower churn rates.
- **Active members churned less:** Customers with frequent activity were more likely to stay.

---

## Feature Engineering
To improve model performance, we:
- **Encoded categorical variables** (e.g., `Geography`, `Gender`).
- **Scaled numerical features** for better model convergence.
- **Addressed class imbalance** by applying **oversampling** and **class weighting** techniques.

---

## Modeling Approach
We trained and compared several models:
- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting (XGBoost)**

Model evaluation was based on:
- **F1 Score** (Primary metric)
- **AUC-ROC Score** (Comparative metric)
- **Accuracy & Precision-Recall Analysis**

---

## Results & Model Performance
| Model                  | F1 Score | AUC-ROC Score | Accuracy |
|------------------------|---------|--------------|----------|
| Logistic Regression    | 0.57    | 0.76         | 0.82     |
| Random Forest         | 0.61    | 0.85         | 0.86     |
| Gradient Boosting (XGB) | **0.65**  | **0.89**  | **0.88** |

 **Gradient Boosting achieved the highest F1 score of 0.65**, exceeding the project requirement of **0.59**.

---

## Key Insights & Business Impact
- **Age and banking activity influence churn:** Older and inactive customers are at higher risk.
- **Active members are more loyal:** Customers who use banking services frequently are less likely to leave.
- **Higher credit scores reduce churn risk:** Customers with good credit ratings tend to stay.

### **Business Recommendations**
- **Implement targeted promotions** for older and inactive customers.
- **Enhance customer engagement** by encouraging product usage.
- **Offer personalized incentives** based on churn risk factors.

---

## Next Steps & Potential Improvements
- **Experiment with deep learning models** for improved accuracy.
- **Analyze customer sentiment** from service interactions.
- **Develop a real-time churn prediction system** for proactive interventions.

---

## How to Use
### Clone the repository:

    git clone https://github.com/rhi-222/bank-churn-prediction.git

### Install dependencies:

    pip install pandas numpy scikit-learn xgboost
    
### Run the Jupyter Notebook:
- Open `[updated]BankChurn(SL).ipynb` in Jupyter Notebook or Google Colab.
- Execute the notebook to preprocess data, train models, and evaluate results.

## Connect With Me
- Email: rhiannon.filli@gmail.com
- LinkedIn: linkedin.com/in/rhiannonfilli
