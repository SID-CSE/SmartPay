# Internship-Project 
# IBM-Skillsbuild--Edunet-Foundation--AICTE
# 💼 Employee Salary Prediction

Welcome to the **Employee Salary Predictor** project! This machine learning-powered app estimates an employee's salary based on their personal and professional attributes like age, experience, education, gender, and occupation. It provides a regression-based approach to predict the **exact salary value**.

---

## 🚀 Project Demo

👉 Try the Live Demo: https://smartpay4.streamlit.app/

📽️ Demo Video: Coming Soon  
📂 Model File: best_salary_regressor.pkl

---

## 📌 Features

- Predict exact salary based on inputs
- Regression model using XGBoost
- Preprocessing pipeline (Numerical + Categorical handling)
- Streamlit UI with dark mode and responsive layout
- CSV Batch prediction support
- Evaluation Metrics: R² Score, RMSE, Actual vs Predicted Plot
- Model performance summary table
- Visual insights for feature importance and predictions

---

## 🧠 Tech Stack & Tools

- Python
- Pandas, Scikit-learn, XGBoost
- Streamlit – App UI
- Matplotlib / Seaborn – Data Visualization
- Joblib – Model Persistence

---

## 📊 Model & Dataset

- Target Variable: Employee Salary (Regression)
- Features Used:
  - Age
  - Experience
  - Gender
  - Education
  - Occupation

- Model: XGBoost Regressor
- Pipeline: Includes preprocessing steps for encoding categorical variables and scaling numeric ones.

---

## 📁 Project Structure

.
├── app.py                        # Streamlit application  
├── best_salary_regressor.pkl    # Saved XGBoost model with preprocessing  
├── model_performance_summary.csv # Metrics summary  
├── input_sample.csv             # Example CSV for batch prediction  
├── README.md                    # Project documentation  
└── requirements.txt             # Required Python packages  

---

## ▶️ Run the App Locally

1. Clone the Repository:
   git clone https://github.com/your-username/employee-salary-prediction.git
   cd employee-salary-prediction

2. Create Virtual Environment:
   python -m venv venv  
   source venv/bin/activate  (or `venv\Scripts\activate` on Windows)

3. Install Dependencies:
   pip install -r requirements.txt

4. Run the Streamlit App:
   streamlit run app.py

---

## 🧪 Model Evaluation Metrics

| Metric     | Value   |
|------------|---------|
| R² Score   | 0.89+   |
| RMSE       | ~2500   |
| MAE        | ~1800   |

Note: Actual performance may vary based on dataset used.

---

## 📦 Batch Prediction (CSV)

- Upload your CSV file via the app to predict salaries in bulk.
- Required columns: Age, Experience, Gender, Education, Occupation

---

## 🙋‍♂️ Author

**Siddharth Kumar**   
LinkedIn: https://linkedin.com/in/siddharthkumaryo

---

## 📄 Development

This project is developed under the Internship through AICTE - IBM SkillsBuild in collaboration with Edunet Foundation

---

## 💡 Future Enhancements

- Add salary trend graphs by age, experience
- Expand dataset for better accuracy
- Deploy on cloud (Streamlit Cloud, AWS, etc.)
- Add authentication for multi-user access

---

## 🙌 Support

If you found this project useful, feel free to ⭐ the repository and share it!
