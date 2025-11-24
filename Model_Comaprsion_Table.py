# -----------------------------------------------------------
#                 IMPORTS
# -----------------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# -----------------------------------------------------------
#                 LOAD DATA
# -----------------------------------------------------------

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# -----------------------------------------------------------
#                 EDA SECTION
# -----------------------------------------------------------

print("\n===== BASIC INFO =====")
print(train.head())
print("Shape:", train.shape)

print("\n===== MISSING VALUES =====")
print(train.isnull().sum())   # We only check, no imputation

print("\n===== SUMMARY STATISTICS =====")
print(train.describe())

print("\n===== TARGET DISTRIBUTION =====")
print(train['y'].value_counts())
print(train['y'].value_counts(normalize=True)*100)

# Correlation heatmap (numerical-only)
numerical_cols = ['age','balance','day','duration','campaign','pdays','previous']
plt.figure(figsize=(10,6))
sns.heatmap(train[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Example distributions
sns.histplot(train['age'], kde=True)
plt.title("Age Distribution")
plt.show()

sns.boxplot(x=train['y'], y=train['duration'])
plt.title("Duration vs Target")
plt.show()

# -----------------------------------------------------------
#                 PREPROCESSING
# -----------------------------------------------------------

x_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]
x_test = test.copy()

categorical_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']

CT = ColumnTransformer([
    ("encoder", OneHotEncoder(), categorical_cols)
], remainder='passthrough')

x_train = CT.fit_transform(x_train)
x_test = CT.transform(x_test)

# Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# -----------------------------------------------------------
#                 MODEL 1: LOGISTIC REGRESSION
# -----------------------------------------------------------

LR_model = LogisticRegression(max_iter=500)
LR_model.fit(x_train, y_train)
LR_pred_train = LR_model.predict(x_train)
LR_pred_test = LR_model.predict(x_test)

LR_accuracy = accuracy_score(y_train, LR_pred_train)
LR_precision = precision_score(y_train, LR_pred_train)
LR_recall = recall_score(y_train, LR_pred_train)
LR_f1 = f1_score(y_train, LR_pred_train)

print("\n===== LOGISTIC REGRESSION RESULTS =====")
print(f"Accuracy:  {LR_accuracy*100:.2f}%")
print(f"Precision: {LR_precision*100:.2f}%")
print(f"Recall:    {LR_recall*100:.2f}%")
print(f"F1 Score:  {LR_f1*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_train, LR_pred_train))

# Confusion Matrix (LR)
cm_lr = confusion_matrix(y_train, LR_pred_train)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# -----------------------------------------------------------
#                 MODEL 2: SVM
# -----------------------------------------------------------

print("\n=== TRAINING OPTIMIZED LINEAR SVM ===")
start_time = time.time()

# Use LinearSVC - much faster for large datasets
SVM_model = LinearSVC(
    random_state=42,
    max_iter=1000,
    dual=False,  # Use primal for n_samples > n_features
    verbose=1,   # Show progress
    C=1.0        # Regularization parameter
)

SVM_model.fit(x_train, y_train)
training_time = time.time() - start_time

print(f"Training completed in: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# Predictions
SVM_pred_train = SVM_model.predict(x_train)

# Metrics
SVM_accuracy = accuracy_score(y_train, SVM_pred_train)
SVM_precision = precision_score(y_train, SVM_pred_train)
SVM_recall = recall_score(y_train, SVM_pred_train)
SVM_f1 = f1_score(y_train, SVM_pred_train)

print("\n===== LINEAR SVM RESULTS =====")
print(f"Accuracy:  {SVM_accuracy*100:.2f}%")
print(f"Precision: {SVM_precision*100:.2f}%")
print(f"Recall:    {SVM_recall*100:.2f}%")
print(f"F1 Score:  {SVM_f1*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_train, SVM_pred_train))

# Confusion Matrix
cm_svm = confusion_matrix(y_train, SVM_pred_train)
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens")
plt.title("Linear SVM - Confusion Matrix")
plt.show()


# -----------------------------------------------------------
#                 MODEL COMPARISON TABLE
# -----------------------------------------------------------

results = {
    "Model": ["Logistic Regression", "SVM (Linear)"],
    "Accuracy": [LR_accuracy, SVM_accuracy],
    "Precision": [LR_precision, SVM_precision],
    "Recall": [LR_recall, SVM_recall],
    "F1 Score": [LR_f1, SVM_f1],
}

df_results = pd.DataFrame(results)
print("\n===== MODEL COMPARISON =====")
print(df_results)
