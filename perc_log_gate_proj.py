import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

data = [[0, 0], [0, 1], [1, 0], [1, 1]]
labelsAND = [0, 0, 0, 1]
labelsOR = [0, 1, 1, 1]

plt.scatter( [x[0] for x in data], labelsAND, c = labelsAND)
plt.show()
# plt.clf()

classifier = Perceptron(max_iter = 40, random_state = 22)
classifier.fit(data, labelsAND)

print(classifier.score(data, labelsAND))

classifier.fit(data, labelsOR)
print(classifier.score(data, labelsOR))

print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))

x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)

point_grid = list(product(x_values, y_values))

distances = classifier.decision_function(point_grid)
abs_distances = abs(distances)

distance_matrix = np.reshape(abs_distances, (100, 100))

plt.pcolormesh(x_values, y_values, distance_matrix)
plt.show()