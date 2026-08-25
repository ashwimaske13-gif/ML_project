# 📊 Customer Service Satisfaction Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-orange)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

Customer satisfaction is a critical performance indicator for e-commerce and customer-support organizations. A large volume of customer support interactions generates valuable information about customer issues, response times, support channels, agent performance, and service outcomes.

This project analyzes customer support data from an e-commerce environment and applies **Exploratory Data Analysis (EDA), statistical analysis, data preprocessing, and Machine Learning classification techniques** to understand the factors associated with customer satisfaction.

The primary objective is to build a machine learning system capable of predicting the **Customer Satisfaction (CSAT) Score** based on customer-support interaction attributes.

---

## 🎯 Problem Statement

In a highly competitive e-commerce environment, maintaining high customer satisfaction is essential for customer retention and business growth.

Customer support teams handle thousands of interactions across different channels, categories, agents, shifts, and issue types. However, raw customer-support data does not directly provide actionable information about which factors influence customer satisfaction.

Therefore, this project aims to:

* Analyze customer-support interaction data.
* Identify important patterns and relationships in customer satisfaction.
* Understand factors influencing CSAT scores.
* Perform statistical analysis to validate business hypotheses.
* Build machine learning classification models.
* Compare different classification algorithms.
* Optimize model performance using hyperparameter tuning.
* Generate insights that can support customer-service improvement.

---

## 💡 Business Objective

The project can help an e-commerce organization answer questions such as:

* Which customer-support channels receive the highest number of tickets?
* Which issue categories are most common?
* Which customer-support factors are associated with satisfaction?
* Does response time influence customer satisfaction?
* Which agent shifts handle the highest ticket volume?
* Which product categories generate more customer-support interactions?
* Can customer satisfaction be predicted automatically?
* How can customer-service operations be improved using data?

---

## 📂 Dataset

The project uses customer-support interaction data containing approximately **85,907 records**.

The dataset contains information related to:

| Feature                   | Description                                   |
| ------------------------- | --------------------------------------------- |
| `Unique id`               | Unique identifier for the support interaction |
| `channel_name`            | Customer-support communication channel        |
| `category`                | Main issue category                           |
| `Sub-category`            | Detailed issue classification                 |
| `Customer Remarks`        | Customer-provided comments                    |
| `Order_id`                | Associated order identifier                   |
| `order_date_time`         | Order timestamp                               |
| `Issue_reported at`       | Time when the issue was reported              |
| `issue_responded`         | Time when the issue was responded to          |
| `Survey_response_Date`    | Customer survey response date                 |
| `Customer_City`           | Customer location                             |
| `Product_category`        | Product category                              |
| `Item_price`              | Product price                                 |
| `connected_handling_time` | Customer-support handling time                |
| `Agent_name`              | Support agent                                 |
| `Supervisor`              | Agent supervisor                              |
| `Manager`                 | Responsible manager                           |
| `Tenure Bucket`           | Agent experience category                     |
| `Agent Shift`             | Support shift                                 |
| `CSAT Score`              | Customer satisfaction score                   |

The notebook shows that the dataset contains **85,907 observations** and 20 columns before preprocessing.

---

# 🔎 Project Workflow

The project follows an end-to-end data science workflow:

```text
Raw Customer Support Data
          ↓
Data Understanding
          ↓
Data Cleaning
          ↓
Missing Value Analysis
          ↓
Exploratory Data Analysis
          ↓
Univariate Analysis
          ↓
Bivariate Analysis
          ↓
Multivariate Analysis
          ↓
Statistical Hypothesis Testing
          ↓
Feature Engineering
          ↓
Encoding & Scaling
          ↓
Correlation Analysis
          ↓
Dimensionality Reduction
          ↓
Train-Test Split
          ↓
Machine Learning Models
          ↓
Model Evaluation
          ↓
Hyperparameter Tuning
          ↓
Final Model Analysis
          ↓
Business Insights
```

---

# 🧹 Data Preprocessing

The dataset was prepared for machine learning through several preprocessing steps.

### Major preprocessing activities

* Missing-value analysis
* Duplicate analysis
* Data type conversion
* Date/time feature processing
* Categorical variable encoding
* Feature selection
* Correlation analysis
* Highly correlated feature removal
* Feature scaling
* Train-test splitting

Highly correlated variables were analyzed and redundant features were removed where appropriate.

Categorical variables were transformed into numerical representations to make them suitable for machine learning algorithms.

---

# 📊 Exploratory Data Analysis

Extensive exploratory data analysis was performed to understand customer-support behavior.

The analysis includes:

### Univariate Analysis

* Ticket distribution
* Customer-support channels
* Issue categories
* Product categories
* Agent shifts
* CSAT score distribution
* Agent tenure

### Bivariate Analysis

Relationships between:

* CSAT and support channel
* CSAT and issue category
* CSAT and product category
* Response time and CSAT
* Agent shift and ticket volume
* Product category and response time

### Multivariate Analysis

Multiple features were analyzed together to identify patterns and relationships influencing customer satisfaction.

Visualization techniques included:

* Bar charts
* Histograms
* Count plots
* Box plots
* Scatter plots
* Correlation heatmaps
* Distribution plots

---

# 🧪 Statistical Analysis

Statistical hypothesis testing was also performed to validate relationships observed during exploratory analysis.

For example, **Welch's independent two-sample t-test** was used to compare response times between Electronics and other product categories.

Welch's t-test was selected because the two groups may have unequal variances.

This statistical approach helps distinguish meaningful differences from patterns that could occur by chance.

---

# 🤖 Machine Learning

The project treats **CSAT Score prediction as a multi-class classification problem**.

The target variable is:

```text
CSAT Score
```

The models implemented include:

### 1. Random Forest Classifier

Random Forest was used as a tree-based ensemble classification algorithm capable of capturing nonlinear relationships between features.

The baseline Random Forest model achieved approximately:

**Accuracy: 95.48%**

However, accuracy alone should not be interpreted as the complete picture because the CSAT classes are imbalanced. Some minority classes have substantially lower recall.

---

### 2. Random Forest + GridSearchCV

Hyperparameter tuning was performed using `GridSearchCV`.

The search included parameters such as:

```text
n_estimators
max_depth
min_samples_split
```

The optimized Random Forest model achieved approximately:

**Accuracy: 95%**

The model showed strong overall performance, although class-level metrics remain important because of the imbalanced target distribution.

---

### 3. Logistic Regression

Logistic Regression was implemented as a baseline linear classification model.

The default Logistic Regression model achieved approximately:

**Accuracy: 43.61%**

The classification report showed weaker performance for several minority classes.

---

### 4. Logistic Regression + GridSearchCV

GridSearchCV was also applied to Logistic Regression.

The hyperparameter search included:

```text
C
penalty
solver
max_iter
```

The best parameters identified were:

```text
C = 0.01
max_iter = 500
penalty = l2
solver = lbfgs
```

The tuned Logistic Regression model achieved approximately:

**Accuracy: 43.61%**

---

# 📈 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

For an imbalanced classification problem, precision, recall, and F1-score are particularly important because overall accuracy can be dominated by the majority class.

### Model Comparison

| Model                     | Accuracy |
| ------------------------- | -------: |
| Random Forest             |  ~95.48% |
| Tuned Random Forest       |     ~95% |
| Logistic Regression       |  ~43.61% |
| Tuned Logistic Regression |  ~43.61% |

The Random Forest model performed substantially better than Logistic Regression on overall accuracy.

---

# ⚠️ Important Model Observation

Although Random Forest achieved approximately **95% accuracy**, the class-level results reveal that performance is not equally strong across all CSAT classes.

For example, the Random Forest classification report shows very strong performance on the majority class but substantially weaker recall for some minority classes.

This indicates **class imbalance** in the target variable.

Therefore, a production system should not rely on accuracy alone.

Future improvements could include:

* Class-weighted models
* SMOTE / oversampling
* Stratified cross-validation
* Balanced Random Forest
* XGBoost / LightGBM
* Advanced feature engineering
* Cost-sensitive learning
* Macro-F1 optimization

---

# 💼 Business Insights

The analysis can provide useful insights for customer-service management.

### Customer Service Optimization

Organizations can identify high-volume support categories and allocate resources accordingly.

### Response-Time Improvement

Analyzing response time against CSAT can help identify service delays that may negatively affect customer satisfaction.

### Workforce Planning

Ticket volume by agent shift can help organizations optimize staffing across different time periods.

### Agent Performance

Agent-level and tenure-level analysis can help identify training requirements and operational bottlenecks.

### Product-Level Analysis

Understanding which product categories generate more customer-support interactions can help organizations identify potential product or service issues.

### Predictive Customer Service

A future production implementation could predict customers who are likely to give low CSAT scores and prioritize those interactions for proactive intervention.

---

# 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Statistical Analysis

* SciPy / statistical hypothesis testing

### Development Environment

* Jupyter Notebook
* Google Colab

---

# 📁 Repository Structure

```text
ML_project/
│
├── Flipcart_ml _submission.ipynb
│
├── README.md
│
└── Customer_support_data.csv
```

> Note: The current repository contains the notebook and README. If the CSV is not committed to GitHub, users will need to obtain the dataset separately before running the notebook.

---

# 🚀 How to Run the Project

## 1. Clone the repository

```bash
git clone https://github.com/ashwimaske13-gif/ML_project.git
```

## 2. Navigate to the project

```bash
cd ML_project
```

## 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Flipcart_ml _submission.ipynb
```

Alternatively, the notebook can be opened directly in **Google Colab**.

---

# 📌 Project Highlights

* 85K+ customer-support records analyzed
* End-to-end EDA
* Univariate, bivariate and multivariate analysis
* Statistical hypothesis testing
* Feature engineering
* Categorical encoding
* Feature scaling
* Correlation analysis
* Dimensionality reduction
* Multi-class classification
* Random Forest
* Logistic Regression
* GridSearchCV
* Confusion matrix
* Classification report
* Business-oriented interpretation

---

# 🔮 Future Enhancements

The project can be extended into a production-grade customer-service intelligence platform.

Potential improvements include:

1. Deploy the trained model using FastAPI or Flask.
2. Build an interactive Streamlit dashboard.
3. Add real-time customer-support prediction.
4. Implement sentiment analysis on customer remarks.
5. Use NLP techniques to analyze customer complaints.
6. Apply SMOTE or class-weighted learning for imbalanced classes.
7. Compare Random Forest with XGBoost and LightGBM.
8. Add MLflow for experiment tracking.
9. Containerize the application using Docker.
10. Deploy the solution on AWS.
11. Add model monitoring and performance tracking.
12. Build a customer-service analytics dashboard.

---

# 🎯 Key Takeaway

This project demonstrates an end-to-end application of **Data Science and Machine Learning to customer-service analytics**.

It combines exploratory analysis, statistical reasoning, feature engineering, classification algorithms, model evaluation, and hyperparameter optimization to understand and predict customer satisfaction.

The project demonstrates how customer-support data can be transformed into actionable insights that can help organizations improve service quality, workforce planning, response times, and overall customer experience.

---

# 👩‍💻 Author

**Ashwini Maske**

Data Scientist | Machine Learning | Python | SQL | Data Analytics

GitHub:
https://github.com/ashwimaske13-gif

---

## ⭐ If you find this project useful

Feel free to ⭐ star the repository and explore the notebook.
