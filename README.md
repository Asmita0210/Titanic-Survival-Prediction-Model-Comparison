# Titanic Survival Prediction Model Comparison

This project demonstrates EDA and data cleaning on the [Kaggle Titanic - train.csv]("https://www.kaggle.com/c/titanic/data?utm_source=chatgpt.com&select=train.csv") dataset, followed by model benchmarking using **Scikit Learn**. We train and compare the prediction accuracy of Logistic Regression, Decision Tree, and Random Forest models to determine the optimal survival classifier.

---

## Project Overview

The objective of this project is to build and evaluate predictive models to classify whether a passenger survived (1) or did not survive (0) the Titanic disaster.  

This project covers:

* **Exploratory Data Analysis (EDA)**: Visualizing missing values, class balances, and feature correlations.
* **Data Cleaning & Imputation**: Handling missing values in Age and Embarked.
* **Feature Engineering**: Removing irrelevant features and applying One-Hot Encoding.
* **Model Training & Evaluation**: Training *Logistic Regression*, *Decision Tree*, and *Random Forest classifiers*, followed by a comparative performance analysis.

---

## Features & Dataset

**Target Variable**
* *Survived*: Survival status (0 = No, 1 = Yes)

**Predictive Features Used**
* *Pclass*: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
* *Sex*: Passenger gender
* *Age*: Age in years
* *SibSp*: Number of siblings/spouses aboard
* *Parch*: Number of parents/children aboard
* *Fare*: Passenger fare
* *Embarked:* Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

**Note**: Columns PassengerId, Name, Ticket, and Cabin were dropped prior to model training to avoid noise.

---

## Installation & Requirements

Ensure you have Python installed along with the required libraries:

<code>pip install pandas numpy matplotlib seaborn scikit-learn</code>

## How To Run

1. **Clone the Repository:**

   <code>git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction</code>

2. **Download the Dataset:**
   Download train.csv from the [Kaggle Titanic Competition]("https://www.kaggle.com/c/titanic/data?utm_source=chatgpt.com&select=train.csv") and place it in the project root directory.

3. **Run the Script or Notebook:**
   <code>python main.py</code>

---

## Model Evaluation & Results

The models were evaluated on a 20% test split (random_state=42) using standard classification metrics:  

|**Model**|**Accurracy**|
|---|---|
|*Logistic Regression*|~81%|
|*Decision Tree*|~79%|
|*Random Forest*|~84% (Best Model)|

*Data subject to variations depending on system factors and network deoendencies*

### Key Classification Metrics (Logistic Regression)

* *Accuracy*: 80% – 83%
* *Precision*: 78% – 82%
* *Recall*: 75% – 80%
* *F1-Score*: 77% – 81%

---

## Project Workflow

1. **Data Loading & Inspection**: Checked basic summary statistics and null values.

   *Null Counts*
   |**COLUMN NAME**|**NULL COUNT**|
   |---|---|
   |Age|177|
   |Cabin|687|
   |Embarked|2|

    <img src="visualisations/Missing Values Heatmap.png" alt="Missing Values Heatmap" width="600"/>


3. **Exploratory Data Analysis (EDA)**: Created count plots for Survived, Sex vs Survived, Pclass vs Survived, and missing value heatmaps.

    <img src="visualisations/Survival Distribution.png" alt=" Survival Distribution" width="600"/>

   Obervation: More passengers died than survived.

    <img src="visualisations/Gender vs Survival.png" alt="Gender vs Survival" width="600"/>

    Observation: Females had much higher survival rates.

    <img src="visualisations/Passenger Class vs Survival.png" alt="Passenger Class vs Survival" width="600"/>

    Observation: First-class passengers survived more frequently.
  
4. **Data Preprocessing**:
   * Imputed Age missing values using the median.
   * Imputed Embarked missing values using the mode.
   * One-hot encoded categorical variables using pd.get_dummies().

---
  
## **Machine Learning Pipeline**

1. **Data Ingestion & Directory Structuring**
   * **Repository Architecture**: Data is loaded dynamically or using relative paths (../data/train.csv) from the root directory to maintain clean separation between raw datasets and execution notebooks.
   * **Dataset Splitting**: The original dataset (df) is split into Training (80%) and Testing (20%) sets before performing feature engineering or scaling. This guarantees that evaluation happens strictly on untouched, unseen real-world samples.

2. **Preprocessing & Feature Engineering**
   * **Baseline Pipeline**: Standard cleaning (missing value imputation, categorical encoding via one-hot encoding, and dropping non-predictive features) is performed to establish baseline datasets (*X_base, y_base*).
   * **Custom/Fabricated Pipeline (df2)**: Advanced feature engineering (such as group-based age imputations or synthetic data generation) is applied strictly to the training portion.
   * **Leakage Prevention**: To avoid test set contamination (data leakage), any rows belonging to the test split are strictly removed from df2 prior to model training.

3. **Model Training & Evaluation**
   Multiple classification algorithms are fitted on the training set:
  * Logistic Regression: Serves as a linear baseline.
  * Decision Tree & Random Forest: Evaluated for non-linear pattern recognition and ensemble decision-making.

4. **Evaluation Metrics & Confusion Matrix Analysis**
   While overall Accuracy Score measures the proportion of total correct predictions:
   $$\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Samples}}$$
   A Confusion Matrix provides a deeper look into model errors by breaking down predictions into four quadrants:
   ||**Predicted Negative (0)**|**Predicted Positive (1)**|
   |---|---|---|
   |Actual Negative (0)|Actual Negative (0)<br>(Correctly predicted Did Not Survive)|False Positive (FP)<br>(Type I Error: Incorrectly predicted Survived)|
   |Actual Positive (1)|False Negative (FN)<br>(Type II Error: Incorrectly predicted Did Not Survive)|True Positive (TP)<br>(Correctly predicted Survived)|

**Key Insights from Evaluation Metric**
   * **Expected Logistic Regression Results:**
     
      Typical Output:
         
      |**Metric**|**Value**|
      |---|---|
      |*Accuracy*|80%-83%|
      |*Precision*|78%-80%|
      |*Recall*|75%-80%|
      |*F1 Score*|77%-81%|

   * **Comparing Models:**

      |**Model**|**Accuracy**|
      |---|---|
      |*Logistic Regression*|0.81|
      |*Decision Tree*|0.79|
      |*Random Forest*|0.84|

   * **Best Model:**
      The expected best model of the three aforementioned model: **Random Forest**

     
 
    
