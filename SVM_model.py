import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score , precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Split features and target
x_train = train.iloc[:, :-1]   # all columns except the last one
y_train = train.iloc[:, -1]    # last column (label)

x_test = test.copy()         

# Identify column types
numerical_cols = ['id', 'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Preprocessing pipeline
CT = ColumnTransformer([
    ("encoder", OneHotEncoder(), categorical_cols)
], remainder='passthrough')


x_train = CT.fit_transform(x_train)
x_test = CT.transform(x_test)

# x_train = pd.DataFrame(x_train)
# x_test = pd.DataFrame(x_test)
# x_train=x_train.iloc[:,1:]
# x_test=x_test.iloc[:,1:]
#print(x_train.head())
#print(x_test.head())

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)

# Train model
SVM_Classifier = SVC(kernel='linear')
SVM_Classifier.fit(x_train,y_train)

test_pred = SVM_Classifier.predict(x_test)

accuracy = accuracy_score(y_train, SVM_Classifier.predict(x_train))
print(f"Training Accuracy: {accuracy * 100:.2f}%")

precision = precision_score(y_train, SVM_Classifier.predict(x_train))
print(f"Training Precision: {precision * 100:.2f}%")    

recall = recall_score(y_train, SVM_Classifier.predict(x_train))
print(f"Training Recall: {recall * 100:.2f}%")  

f1 = f1_score(y_train, SVM_Classifier.predict(x_train))
print(f"Training F1 Score: {f1 * 100:.2f}%")    

print("\nClassification Report:")
print(classification_report(y_train, SVM_Classifier.predict(x_train)))
