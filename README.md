# 🏦 Loan Approval Prediction

This project predicts whether a loan application will be approved or rejected using machine learning techniques. It includes data preprocessing, class balancing using **SMOTE**, and modeling with **Logistic Regression** and **Decision Tree** classifiers.

---

## 📌 Objective

- Analyze loan application data.
- Preprocess and encode categorical variables.
- Handle imbalanced classes using SMOTE.
- Train models using Logistic Regression and Decision Tree.
- Evaluate models using classification metrics and confusion matrices.

---

## 🧰 Libraries Used

```python
# Data manipulation and visualization
pandas, numpy, matplotlib, seaborn

# Machine learning
scikit-learn (LogisticRegression, DecisionTreeClassifier, LabelEncoder, train_test_split, metrics)
imblearn (SMOTE)

##📂 Dataset
#loan_approval_dataset.csv
Pre-cleaned dataset with features like education, employment status, income, etc.
Target variable: loan_status (Approved or Rejected)

##⚙️ Workflow
1-Data Preprocessing
-Strip extra spaces from column names
-Drop irrelevant loan_id column
-Encode categorical features (education, self_employed, loan_status)
-Scale features using StandardScaler (for logistic regression)
2-Data Exploration
-Visualize target class imbalance
-Generate correlation heatmap
3-Train-Test Split
-Use stratified 80-20 split to preserve class distribution
4-Class Balancing with SMOTE
-Oversample minority class in training data
5-Model Training
-Train Logistic Regression and Decision Tree models on balanced data
6-Evaluation
-Use classification_report for precision, recall, F1-score
-Plot confusion matrices for both models

##📈 Results
🔹 Logistic Regression
-Trained on SMOTE-balanced data
-Evaluated on unbalanced test set
🔹 Decision Tree
-Trained and evaluated similarly
#Evaluation Metrics Include:
-Accuracy
-Precision
-Recall
-F1-score
#Visualizations:
-Class distribution bar plot
-Correlation heatmap
-Confusion matrices for both models

📝 License
This project is licensed under the MIT License.
