import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()

# print(aaron_judge.columns)
# print(aaron_judge.type.unique())

david_ortiz['type'] = david_ortiz.type.map({'S': 1, 'B': 0})
# print(aaron_judge.type.unique())

# print(aaron_judge['plate_x'])

df = david_ortiz.dropna(subset = ['plate_x', 'plate_z', 'type'])
# print(df)

plt.scatter(df.plate_x, df.plate_z, c = df.type, cmap = plt.cm.coolwarm, alpha = 0.4)
plt.legend()
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(df[['plate_x', 'plate_z']], df.type, test_size = 0.2 ,random_state = 1)

classifier = SVC(kernel = 'rbf', gamma = 3, C = 1 )

fitted = classifier.fit(x_train, y_train)

draw_boundary(ax, fitted)
plt.show()

print(classifier.score(x_test, y_test))