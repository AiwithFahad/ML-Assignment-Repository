# 📘 Simple Linear Regression — Student Performance (Assignment 1)

---

## 📌 Overview
This project implements **Simple Linear Regression** to predict student exam scores based on the number of study hours.  

The workflow includes:
- Data loading
- Data preprocessing
- Model training
- Prediction
- Evaluation using regression metrics

The goal is to understand how study time impacts student performance using machine learning.

---

## 📂 Dataset Information

- **Dataset Used:** `StudentsPerformance.csv`
- **Shape:** (1000, 8)
- **Source:** Kaggle Students Performance Dataset
- **Target Variable:** `Exam_Score` (derived from student performance scores)

### 📌 Features Used:
- **Independent Variable (X):** `Study_Hours`
- **Dependent Variable (y):** `Exam_Score`

---

## ⚙️ Processing Steps

1. Data loading and inspection  
2. Data cleaning (no missing values or duplicates)  
3. Feature selection (Study Hours → Exam Score)  
4. Train-test split (80% training, 20% testing)  
5. Model training using **Simple Linear Regression (Scikit-Learn)**  
6. Prediction on test data  
7. Model evaluation using regression metrics  

---

## 📈 Regression Model

### 🧮 Learned Equation


```math
Exam_Score = 43.3039 + 3.4874 × Study_Hours


📊 Model Performance
| Metric   | Value  | Interpretation                            |
| -------- | ------ | ----------------------------------------- |
| R² Score | 0.7970 | Model explains ~79.7% of variance in data |
| RMSE     | 6.5973 | Average prediction error ≈ 6.6 marks      |




🔍 Key Findings
📊 Strong positive correlation between study hours and exam score
📈 Each additional hour of study increases score by ~3.49 marks
🎯 Model explains ~80% of variation in student performance
⚠️ Remaining variance suggests other influencing factors exist



📉 Visualizations Generated
Scatter Plot (Study Hours vs Exam Score)
Regression Line Fit
Residual Plot
Distribution plots
Correlation heatmap



🧾 Conclusion

The Simple Linear Regression model successfully predicts student performance using study hours as the only feature.

While the model performs reasonably well, incorporating additional variables such as attendance, previous academic scores, or study habits could significantly improve prediction accuracy.



🏁 Final Result Summary

Simple Linear Regression — Student Performance (Assignment 1)

Dataset: StudentsPerformance.csv (Shape: 1000, 8)

Regression Equation:
Exam_Score = 43.3039 + 3.4874 × Study_Hours

Model Performance:
R² Score = 0.7970
RMSE     = 6.5973

