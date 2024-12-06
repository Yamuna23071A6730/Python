import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'credit_risk_dataset.csv'
df = pd.read_csv(file_path)

df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)  # Impute with median
df['loan_int_rate'].fillna(df['loan_int_rate'].mean(), inplace=True)  # Impute with mean

df['person_home_ownership'] = df['person_home_ownership'].astype('category').cat.codes
df['loan_intent'] = df['loan_intent'].astype('category').cat.codes
df['loan_grade'] = df['loan_grade'].astype('category').cat.codes
df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

X = df[['person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 
        'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 
        'loan_percent_income', 'cb_person_cred_hist_length']]
y = df['loan_status']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_data = pd.DataFrame({
    'person_age': [30],
    'person_income': [50000],
    'person_home_ownership': [0],  
    'person_emp_length': [5],
    'loan_intent': [1],  
    'loan_grade': [2],  
    'loan_amnt': [20000],
    'loan_int_rate': [12.5],
    'loan_percent_income': [0.4],
    'cb_person_cred_hist_length': [6]
})
new_prediction = model.predict(new_data)
print("\nPrediction for new applicant (0 = No Default, 1 = Default):", new_prediction[0])
