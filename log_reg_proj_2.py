import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns


col_names = ['age', 'workclass', 'fnlwgt','education', 'education-num', 
'marital-status', 'occupation', 'relationship', 'race', 'sex',
'capital-gain','capital-loss', 'hours-per-week','native-country', 'income']
df = pd.read_csv('adult.data',header = None, names = col_names)

#Clean columns by stripping extra whitespace for columns of type "object"
for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].str.strip()
print(df.head())

#1. Check Class Imbalance
df.income = np.where(df.income == '<=50K', 0, 1)

print(df.income.value_counts(normalize = True))

#2. Create feature dataframe X with feature columns and dummy variables for categorical features
feature_cols = ['age','capital-gain', 'capital-loss', 'hours-per-week', 'sex','race', 'hours-per-week', 'education']

X = pd.get_dummies(df[feature_cols], drop_first = True)
#3. Create a heatmap of X data to see feature correlation
sns.heatmap(X.corr())
plt.show()
plt.clf()
#4. Create output variable y which is binary, 0 when income is less than 50k, 1 when it is greater than 50k
y = df.income

#5a. Split data into a train and test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#5b. Fit LR model with sklearn on train set, and predicting on the test set
LR = LogisticRegression(C = 0.05, penalty = "l1", solver = 'liblinear')
model = LR.fit(x_train, y_train)

y_pred = model.predict(x_test)
#6. Print model parameters (intercept and coefficients)
coeff = model.coef_
intercept = model.intercept_

print(f'Model Parameters, Intercept: {intercept}')

print(f'Model Parameters, Coeff:{coeff}')


#7. Evaluate the predictions of the model on the test set. Print the confusion matrix and accuracy score.
cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

print(f'Confusion Matrix on test set: {cm}')
print(f'Accuracy Score on test set: {accuracy}')

# 8.Create new DataFrame of the model coefficients and variable names; sort values based on coefficient
var_coeff = pd.DataFrame(zip(x_train.columns, model.coef_[0]), columns = ['vars', 'coeff']).sort_values('coeff')

var_coeff = var_coeff.drop(var_coeff[var_coeff['coeff'] == 0].index, axis = 0).reset_index(drop = True)

# print(var_coeff)
#9. barplot of the coefficients sorted in ascending order
sns.barplot(data = var_coeff, x = 'vars', y = 'coeff')
plt.xticks(rotation = 90)
plt.show()
plt.clf()
#10. Plot the ROC curve and print the AUC value.
y_pred_prob = LR.predict_proba(x_test)

roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
print(f'ROC AUC score: {roc_auc}')

fpr, tpr, thresholds = roc_curve(y_test,y_pred_prob[:,1])

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0,1],[0,1], color='navy',linestyle='--')
plt.title('ROC Curve')
plt.grid()
plt.show()