import pandas as pd import numpy as np import matplotlib.pyplot as plt from sklearn.model_selection import train_test_split from sklearn.linear_model import LogisticRegression from sklearn.preprocessing import LabelEncoder from sklearn.metrics import roc_curve, auc
#Load the dataset data pd.read_csv('/creditproject.csv')
# Display the first few rows to understand the dataset print(data.head())
#Check if the 'default payment next month' column is present
if 'default payment next month' not in data.columns:
raise ValueError("Column 'default payment next month' not found in the dataset.")
else: #Convert 'default' column to binary values (1 for 'Y' and for 'N')
data['default'] = data['default payment next month'].apply(lambda x: 1 if x 'Y' else 8)
# Preprocess the data data pd.get_dummies (data, columns['EDUCATION', 'MARRIAGE'])
#Handle non-numeric values in the dataset
for col in data.select_dtypes (include=['object']).columns:
le LabelEncoder()
data[col] le.fit_transform(data[col])
#Split the data into features and target variable
X= data.drop('default', axis=1)
y data['default']
#Split the data into train and test sets
X_train, X_test, y_train, y_test train_test_split(X, y, test_size=0.2, random_state=42)
#Train the logistic regression model model LogisticRegression() model.fit(X_train, y_train)
#Predict the probabilities on the test set y_pred_proba model.predict_proba (X_test) [:, 1]
#Calculate the false positive rate and true positive rate fpr, tpr, thresholds roc_curve(y_test, y_pred_proba)
#Calculate the K-S statistic ks_statistic np.max(np.abs(tpr fpr)) print(f"K-S statistic: (ks_statistic:.4f}")
#Plot the K-S chart
plt.figure(figsize=(8, 6))
plt.plot(thresholds, tpr, label-'True Positive Rate')
plt.plot(thresholds, fpr, label='False Positive Rate')
plt.xlabel('Probability Threshold')
plt.ylabel('Rate')
plt.title('K-S Chart')
plt.legend()
plt.show()
#Calculate the AUC
auc_score auc (fpr, tpr)
print(f"AUC: {auc_score:.4f)")
