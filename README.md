# Telecom Customer Churn Prediction

## Project Overview

This project analyzes telecom customer data and builds machine learning models to predict customer churn.

Customer churn means a customer leaves or cancels their service. In a telecom business, predicting churn is useful because the company can identify customers who may leave and take action to retain them.

## Goal

The goal of this project is to:

- Explore telecom customer data
- Understand patterns related to churn
- Build machine learning models to predict churn
- Compare model performance
- Save the best-performing model

## Dataset

The dataset used in this project is the **Customer Churn** dataset from Kaggle.

Dataset source: Kaggle — Customer Churn by Barun Kumar

The dataset contains customer-level telecom information such as:

- Account length
- Contract renewal status
- Data plan status
- Data usage
- Customer service calls
- Daytime minutes and calls
- Monthly charges
- Overage fees
- Roaming minutes
- Churn status

The target variable is:

```text
Churn
```

Where:

```text
0 = customer stayed
1 = customer churned
```

## Project Structure

```text
telecom customer churn prediction/
│
├── data/
│   └── telecom_churn.csv
│
├── models/
│   └── random_forest_churn_model.pkl
│
├── notebooks/
│   └── telecom_churn_analysis.ipynb
│
├── src/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Tools and Libraries

This project uses:

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook

## Workflow

### 1. Data Loading

The dataset was loaded using Pandas.

```python
df = pd.read_csv("../data/telecom_churn.csv")
```

### 2. Data Exploration

I inspected the dataset using:

- `df.head()`
- `df.shape`
- `df.columns`
- `df.info()`
- `df.describe()`

The dataset contains **3,333 customer records** and **11 columns**.

### 3. Missing Value Check

I checked for missing values using:

```python
df.isnull().sum()
```

The dataset had no missing values.

### 4. Churn Distribution

The target column `Churn` was analyzed to understand how many customers stayed versus churned.

Results:

```text
0 = 2,850 customers stayed
1 = 483 customers churned
```

This means the dataset is imbalanced, with more customers staying than leaving.

### 5. Exploratory Data Analysis

Several patterns were explored, including:

- Churn distribution
- Average customer service calls by churn
- Average monthly charge by churn
- Average contract renewal by churn

Key insights:

- Customers who churned made more customer service calls on average.
- Customers who churned had slightly higher monthly charges on average.
- Customers who stayed had a higher contract renewal rate.

### 6. Machine Learning Models

Two classification models were trained and compared:

1. Logistic Regression
2. Random Forest Classifier

The data was split into training and testing sets using an 80/20 split.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 7. Model Evaluation

The models were evaluated using:

- Accuracy
- Confusion matrix
- Precision
- Recall
- F1-score

### Logistic Regression Results

Logistic Regression achieved about **86% accuracy**, but it had low recall for churned customers.

### Random Forest Results

Random Forest achieved about **93% accuracy** and performed better at identifying churned customers.

For class `1` churn customers, Random Forest achieved:

```text
Precision: 0.86
Recall: 0.61
F1-score: 0.72
```

This made Random Forest the better model for this project.

## Final Model

The Random Forest model was saved using Joblib:

```python
joblib.dump(rf_model, "../models/random_forest_churn_model.pkl")
```

Saved model:

```text
models/random_forest_churn_model.pkl
```

## Key Takeaways

- The dataset is imbalanced, with most customers not churning.
- Customer service calls appear to be an important churn-related factor.
- Contract renewal is strongly related to customer retention.
- Random Forest performed better than Logistic Regression for this churn prediction task.

## Future Improvements

Possible future improvements include:

- Add more visualizations
- Tune Random Forest hyperparameters
- Try other models such as Gradient Boosting or XGBoost
- Use cross-validation
- Handle class imbalance with class weights or resampling
- Build a small prediction app using Streamlit or Gradio

## Author

Created as a beginner-friendly data analysis and machine learning portfolio project.
