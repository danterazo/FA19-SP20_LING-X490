# LING-X 490 Assignment 5: Waseem SVM
# Dante Razo, drazo, 10/30/2019
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import sklearn.metrics
import numpy as np
import random

# Import data
# TODO: remove http://t.co/* links
data_dir = "./data/"
a = open(data_dir + "X_train", "r", encoding="utf-8")
X_train = np.asarray(a.read().splitlines()).reshape(-1, 1)

a = open(data_dir + "x_test", "r", encoding="utf-8")
X_test = np.asarray(a.read().splitlines()).reshape(-1, 1)

a = open(data_dir + "y_train", "r", encoding="utf-8")
y_train = a.read().splitlines()
for i in range(0, len(y_train)):
    y_train[i] = int(y_train[i])

a = open(data_dir + "y_test", "r", encoding="utf-8")
y_test = a.read().splitlines()
for i in range(0, len(y_test)):
    y_test[i] = int(y_test[i])

a.close()

# Feature engineering: One-hot encoding
enc = OneHotEncoder(handle_unknown='ignore')
X_train = enc.fit_transform(X_train)
X_test = enc.transform(X_test)

# Shuffle data (keeps indices)
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

# Fitting the model
print("Training...")
svm = SVC(kernel="linear", gamma="auto")
svm.fit(X_train, y_train)
print("Training complete.\n")

""" KERNEL RESULTS
rbf: 0.6844783715012722
linear: 0.6844783715012722
poly: 0.6844783715012722
sigmoid: 0.6844783715012722
precomputed: N/A, not supported
"""

# Testing + results
rand_acc = sklearn.metrics.balanced_accuracy_score(y_test, [random.randint(1, 2) for x in range(0, len(y_test))])
print(f"Random/Baseline Accuracy: {rand_acc}")
print(f"Testing Accuracy: {sklearn.metrics.accuracy_score(y_test, svm.predict(X_test))}")
