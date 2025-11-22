import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

#LOAD DATA
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#EDA SECTION
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

#PREPROCESSING
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


#SVM Model
SVM_model = SVC(kernel='linear')
SVM_model.fit(x_train, y_train)
SVM_pred_train = SVM_model.predict(x_train)
SVM_pred_test = SVM_model.predict(x_test)

SVM_accuracy = accuracy_score(y_train, SVM_pred_train)
SVM_precision = precision_score(y_train, SVM_pred_train)
SVM_recall = recall_score(y_train, SVM_pred_train)
SVM_f1 = f1_score(y_train, SVM_pred_train)

print("\n===== SVM RESULTS =====")
print(f"Accuracy:  {SVM_accuracy*100:.2f}%")
print(f"Precision: {SVM_precision*100:.2f}%")
print(f"Recall:    {SVM_recall*100:.2f}%")
print(f"F1 Score:  {SVM_f1*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_train, SVM_pred_train))

# Confusion Matrix (SVM)
cm_svm = confusion_matrix(y_train, SVM_pred_train)
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens")
plt.title("SVM - Confusion Matrix")
plt.show()
