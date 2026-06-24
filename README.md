# (Assignment 1)
# 📘 Simple Linear Regression — Student Performance (Assignment 1)

---

## 📌 Overview

This project implements Simple Linear Regression to predict student exam scores based on the number of study hours.

The workflow includes:

* Data loading
* Data preprocessing
* Model training
* Prediction
* Evaluation using regression metrics

The goal is to understand how study time impacts student performance using machine learning.

---

## 📂 Dataset Information

* Dataset Used: StudentsPerformance.csv
* Shape: (1000, 8)
* Source: Kaggle Students Performance Dataset
* Target Variable: Exam_Score (derived from student performance scores)

### 📌 Features Used:

* Independent Variable (X): Study_Hours
* Dependent Variable (y): Exam_Score

---

## ⚙️ Processing Steps

1. Data loading and inspection
2. Data cleaning (no missing values or duplicates)
3. Feature selection (Study Hours → Exam Score)
4. Train-test split (80% training, 20% testing)
5. Model training using Simple Linear Regression (Scikit-Learn)
6. Prediction on test data
7. Model evaluation using regression metrics

---

## 📈 Regression Model

### 🧮 Learned Equation

Exam_Score = 43.3039 + 3.4874 × Study_Hours

---

## 📊 Model Performance

Metric: R² Score
Value: 0.7970
Interpretation: Model explains ~79.7% of variance in data

Metric: RMSE
Value: 6.5973
Interpretation: Average prediction error ≈ 6.6 marks

---

## 🔍 Key Findings

* Strong positive correlation between study hours and exam score
* Each additional hour of study increases score by ~3.49 marks
* Model explains ~80% of variation in student performance
* Remaining variance suggests other influencing factors exist

---

## 📉 Visualizations Generated

* Scatter Plot (Study Hours vs Exam Score)
* Regression Line Fit
* Residual Plot
* Distribution plots
* Correlation heatmap

---

## 🧾 Conclusion

The Simple Linear Regression model successfully predicts student performance using study hours as the only feature.

While the model performs reasonably well, incorporating additional variables such as attendance, previous academic scores, or study habits could significantly improve prediction accuracy.

---

## 🏁 Final Result Summary

Simple Linear Regression — Student Performance (Assignment 1)

Dataset: StudentsPerformance.csv (Shape: 1000, 8)

Regression Equation:
Exam_Score = 43.3039 + 3.4874 × Study_Hours

Model Performance:
R² Score = 0.7970
RMSE = 6.5973




----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# (Assignment 2)
# 📘 Multivariate Linear Regression — Medical Insurance Price Prediction 

---

## 📌 Overview

This project implements **Multivariate Linear Regression** to predict **medical insurance charges** using multiple independent variables.

The workflow includes:

* Data loading
* Exploratory Data Analysis (EDA)
* Data preprocessing and encoding
* Feature selection
* Train-test splitting
* Model training
* Prediction
* Regression equation analysis
* Model evaluation using regression metrics
* Manual verification of predictions
* Feature comparison experiments

The goal of this assignment is to understand how multiple factors such as **age, BMI, number of children, smoking status, sex, and region** affect medical insurance charges using machine learning.

---

## 📂 Dataset Information

* **Dataset Used:** Medical_insurance.csv
* **Dataset Path:** `/content/drive/MyDrive/Medical_insurance.csv`
* **Shape:** (2772, 7)
* **Source:** Medical Insurance Price Prediction Dataset
* **Target Variable:** `charges`

---

## 🧾 Features Used

The dataset contains the following input features:

* **age** — age of the individual
* **sex** — gender of the individual
* **bmi** — Body Mass Index
* **children** — number of dependents covered by insurance
* **smoker** — whether the individual is a smoker or not
* **region** — residential region of the individual

### 🎯 Target Variable

* **charges** — medical insurance cost to be predicted

---

## 🛠 Tasks Performed

### 1) Data Loading

* Imported all required Python libraries
* Loaded the **Medical_insurance.csv** dataset into a dataframe

### 2) Dataset Understanding / EDA

Performed a full exploratory analysis of the dataset, including:

* Checking dataset shape
* Inspecting column names and data types
* Reviewing summary statistics
* Checking for missing values
* Exploring unique values in categorical columns
* Visualizing the distribution of insurance charges
* Exploring relationships between features and charges using:

  * scatter plots
  * box plots

### 3) Data Preprocessing

Prepared the dataset for regression modeling by:

* Creating a working copy of the original dataframe
* Encoding binary categorical features:

  * `sex`
  * `smoker`
* Encoding the multi-category feature:

  * `region`
* Ensuring all features were in a machine-learning-ready format

### 4) Feature Selection and Train-Test Split

* Defined **X** as the set of independent variables
* Defined **y** as the target variable (`charges`)
* Split the dataset into:

  * **80% training data**
  * **20% testing data**

### 5) Model Training

* Trained a **Linear Regression** model using the prepared training data
* Learned regression coefficients for each feature
* Learned the model intercept

### 6) Coefficients and Regression Equation Analysis

* Displayed the coefficients of all selected features
* Displayed the intercept value
* Printed the complete regression equation
* Interpreted how each feature affects insurance charges

### 7) Model Evaluation

Evaluated model performance using the following regression metrics:

* **R² Score**
* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**

Also analyzed performance through visualizations such as:

* **Actual vs Predicted Charges**
* **Residual plots**

### 8) Manual Verification

* Selected one test sample
* Manually calculated the predicted insurance charge using the regression equation
* Compared the manual result with the model prediction to verify the model logic

---

## 🧪 Practice Tasks / Experiments

### Task A — Full Model

Built a multivariate regression model using all relevant available features.

### Task B — Numeric-Only Model

Built a model using only numeric variables:

* `age`
* `bmi`
* `children`

### Task C — Numeric + Encoded Categorical Variables

Built a model using numeric features together with encoded categorical variables.

### Task D — Feature Comparison Experiment

Compared multiple models using different feature subsets to study the impact of feature selection on model performance.

### Task E — Most Influential Feature Analysis

Analyzed which feature had the strongest impact on insurance charges based on:

* coefficient magnitude
* individual predictive power

---

## 📈 Extra Practice Ideas

Additional experiments included:

* Building a **smoker-only model**
* Applying **feature scaling**
* Checking **correlation among numeric variables**
* Removing one feature and retraining the model
* Comparing different feature combinations for better understanding

---

## 📊 Outcome / Results

The assignment successfully demonstrates the application of **Multivariate Linear Regression** on a real-world medical insurance dataset.

The **best-performing model** used the following features:

* `age`
* `bmi`
* `children`
* `sex_encoded`
* `smoker_encoded`

This model achieved an **R² Score of approximately 0.74 on the test set**.




