# Telco Customer Churn Analytics Dashboard

## 📌 Project Overview
An **interactive analytics dashboard** built with **Python, Streamlit, and Machine Learning** to analyze and predict customer churn for a telecom company. Designed to empower business users and data analysts to explore churn risk, financial impact, and actionable insights through intuitive visualizations and filtering.


---

## 🎯 Key Features
- **End-to-End Pipeline** – Load, clean, analyze, and visualize data in one streamlined tool
- **Churn Prediction** – XGBoost classification model providing churn probability and risk segmentation 
- **Dynamic Filtering** – Filter by churn risk, internet service type, and contract type
- **Financial Impact Analysis** – Revenue at risk, customer lifetime value (CLTV), and tenure insights
- **Customer Lookup** – Search individual customers by ID for detailed churn profiles  
- **Model Performance** – Confusion matrix, ROC curve, precision, recall, and F1-score visualizations
- **Business Recommendations** – Data-driven strategies for targeted retention and reducing churn  

---

## 📊 Key Metrics
- **Churn Rate:** Percentage of customers who have churned
- **High Risk Customers:** Number of customers flagged as high churn risk
- **Revenue at Risk:** Monthly revenue at risk from high-risk customers
- **CLTV:** Average lifetime value of customers
- **Customer Tenure:** Average customer relationship duration

---

## 🛠 Tech Stack
- **Language:** Python  
- **Libraries:** pandas, numpy, plotly, scikit-learn, xgboost  
- **Visualization:** Streamlit, Plotly  
- **Machine Learning:** XGBoost classification model
- **Dataset:** [Telco Customer Churn (7,043 customers, California, Q3 data)](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3?resource=download)

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/jshchng/telco-churn-dashboard.git
cd telco-churn-dashboard
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```
---

## 📈 Business Impact
This project showcases the ability to:
- Identify and prioritize customers at risk of churn using predictive modeling
- Quantify financial exposure from churn to inform retention budgets
- Segment customers by service type and contract for targeted interventions
- Monitor and improve model performance for operational use
- Communicate complex data insights via a user-friendly interactive dashboard

