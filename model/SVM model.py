# Importing all the modules and packages

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Importing and setting up dataset:

data = pd.read_csv("train.csv")

# Separating independent and dependent variables:

feature_set = data[['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username', 'description length', 'external URL', 'private', '#posts', '#followers', '#follows']]

# Independent variable:
x = np.asarray(feature_set)
print("x:", x)

# Dependent variable:
y = np.asarray(data['fake'])
print("y:", y)

# Splitting data for training and testing:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)

print("x_test:", x_test)
print("x_train:", x_train)
print("y_test:", y_test)
print("y_train:", y_train)
# Trying different kernels:

# Using linear kernel:
classifier1 = svm.SVC(kernel='linear', gamma='auto', C=2, probability=True)
classifier1.fit(x_train, y_train)
y1_predict = classifier1.predict(x_test)

print("Raw Prediction:", y1_predict)

print(classification_report(y_test, y1_predict))

# # Using radial basis function kernel:
# classifier2 = svm.SVC(kernel='rbf', gamma='auto', C=2)
# classifier2.fit(x_train, y_train)
# y2_predict = classifier2.predict(x_test)
#
# print(classification_report(y_test, y2_predict))
#
# # Using sigmoid kernel:
# classifier3 = svm.SVC(kernel='sigmoid', gamma='auto', C=2)
# classifier3.fit(x_train, y_train)
# y3_predict = classifier3.predict(x_test)
#
# print(classification_report(y_test, y3_predict))

# acc1 = accuracy_score(y_test, y1_predict)
# acc2 = accuracy_score(y_test, y2_predict)
# acc3 = accuracy_score(y_test, y3_predict)

# print(f'Accuracy with linear kernel is {round(acc1*100, 2)}%')
# print(f'Accuracy with rbf kernel is {round(acc2*100, 2)}%')
# print(f'Accuracy with sigmoid kernel is {round(acc3*100, 2)}%')

pickle.dump(classifier1, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))