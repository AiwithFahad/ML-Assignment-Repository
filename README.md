📘 Simple Linear Regression — Student Performance (Assignment 1)
📌 Overview

This project implements Simple Linear Regression to predict student exam scores based on the number of study hours. The workflow includes data loading, preprocessing, model training, prediction, and evaluation using standard regression metrics.

📂 Dataset Information
Dataset Used: StudentsPerformance.csv
Shape: (1000, 8)
Source: Kaggle Students Performance Dataset
Target Variable: Exam_Score (derived from student performance scores)
Feature Used:
Independent Variable (X): Study_Hours
Dependent Variable (y): Exam_Score
⚙️ Processing Steps
Data loading and cleaning
Feature selection (Study Hours → Exam Score)
Train-test split (80% training, 20% testing)
Model training using Simple Linear Regression (Scikit-Learn)
Prediction on test data
Model evaluation using regression metrics
📈 Regression Model
🧮 Equation Learned
Exam_Score = 43.3039 + 3.4874 × Study_Hours
📊 Model Performance
Metric	Value	Interpretation
R² Score	0.7970	Model explains ~79.7% of variance
RMSE	6.5973	Average prediction error ≈ 6.6 marks
🔍 Key Findings
📊 Strong positive relationship between study hours and exam score
📈 Each additional study hour increases score by ~3.49 marks
🎯 Model explains ~80% of variation in student performance
⚠️ Some variability exists, suggesting other factors also affect performance
📉 Visualizations (Generated)
Scatter Plot (Study Hours vs Exam Score)
Regression Line Fit
Residual Plot
Distribution plots for variables
Correlation heatmap
🧾 Conclusion

The Simple Linear Regression model successfully predicts student performance based on study hours with reasonable accuracy. While study time is a strong predictor, incorporating additional features (such as attendance, prior grades, or learning habits) could further improve model performance.

🏁 Assignment Complete
Simple Linear Regression — Student Performance
Regression Equation: Exam_Score = 43.3039 + 3.4874 × Study_Hours
R² Score: 0.7970
RMSE: 6.5973
Dataset: StudentsPerformance.csv (Shape: 1000, 8)<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e60cea84-83ff-43fa-b133-6e80d50d8e60" />
