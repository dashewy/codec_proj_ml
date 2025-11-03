import codecademylib3
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


#https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
'red','green','blue','gold','white','black','orange','mainhue','circles',
'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data", names = cols)

#variable names to use as predictors
var = [ 'red', 'green', 'blue','gold', 'white', 'black', 'orange', 'mainhue','bars','stripes', 'circles','crosses', 'saltires','quarters','sunstars','triangle','animate']

#Print number of countries by landmass, or continent
print(df.landmass.value_counts())

#Create a new dataframe with only flags from Europe and Oceania
df_36 = df[df.landmass.isin([3,6])]
print(df_36)

#Print the average vales of the predictors for Europe and Oceania
print(df_36.groupby('landmass')[var].mean())

#Create labels for only Europe and Oceania
labels = df_36["landmass"]

#Print the variable types for the predictors
print(labels.dtypes)

#Create dummy variables for categorical predictors
data = pd.get_dummies(df_36[var])
print(data)
#Split data into a train and test set
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = .4, random_state = 1)
#Fit a decision tree for max_depth values 1-20; save the accuracy score in acc_depth
depths = range(1, 21)
acc_depth = []

for i in depths:
  model = DecisionTreeClassifier(max_depth = i)
  model.fit(x_train, y_train)
  
  acc_depth.append(model.score(x_test, y_test))

# print(acc_depth)
print(acc_depth)
#Plot the accuracy vs depth
plt.plot(depths, acc_depth)
plt.show()
plt.clf()

#Find the largest accuracy and the depth this occurs
best_depth = np.argmax(acc_depth)
print(best_depth)
#Refit decision tree model with the highest accuracy and plot the decision tree
model_best = DecisionTreeClassifier(max_depth = best_depth)
model_best.fit(x_train, y_train)

#Create a new list for the accuracy values of a pruned decision tree.  Loop through
#the values of ccp and append the scores to the list
acc_pruned = []
ccp = np.logspace(-3, 0, num=20)
for i in ccp:
    dt_prune = DecisionTreeClassifier(random_state = 1, max_depth = best_depth, ccp_alpha=i)
    dt_prune.fit(x_train, y_train)
    acc_pruned.append(dt_prune.score(x_test, y_test))



#Plot the accuracy vs ccp_alpha
plt.plot(ccp, acc_pruned)
plt.xscale('log')
plt.show()
plt.clf()

#Find the largest accuracy and the ccp value this occurs
large_ccp_alpha = np.max(acc_pruned)
print(large_ccp_alpha)

#Fit a decision tree model with the values for max_depth and ccp_alpha found above
dt_final = DecisionTreeClassifier(random_state = 1, max_depth = best_depth, ccp_alpha=large_ccp_alpha)
dt_final.fit(x_train, y_train)

#Plot the final decision tree
tree.plot_tree(dt_final, feature_names = x_train.columns,  
               class_names = ['Europe', 'Oceania'],
                filled=True)

#Plot the final decision tree
plt.show()
