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
