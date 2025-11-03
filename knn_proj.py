import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()

# print(breast_cancer_data.data[0])
# print(breast_cancer_data.feature_names)

# print(breast_cancer_data.target)
# print(breast_cancer_data.target_names)

training_data, validation_set, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

# print(len(training_data))
# print(len(training_labels))

classifier = KNeighborsClassifier(n_neighbors = 3)

classifier.fit(training_data, training_labels)

print(classifier.score(validation_set, validation_labels))

score_list = []
for k in range(1,100):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)

  score_list.append(classifier.score(validation_set, validation_labels))

max_value = max(score_list)
max_value_index = score_list.index(max_value)

print(f'max value {max_value}, at index {max_value_index}')

plt.plot(range(1,100), score_list)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()