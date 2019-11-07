# LING-X 490 Assignment 5: Waseem SVM
# Dante Razo, drazo, 10/30/2019
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics
import numpy as np
import random

# Import data
# TODO: remove http://t.co/* links
# IDEA: bag of words
data_dir = "./data/"
a = open(data_dir + "X_train", "r", encoding="utf-8")
X_train_lines = np.asarray(a.read().splitlines()).reshape(-1, 1)
X_train = list()
for i in range(0, X_train_lines.shape[0]):  # bag of words (?)
    X_train.append(X_train_lines[i][0].split())

a = open(data_dir + "x_test", "r", encoding="utf-8")
X_test_lines = np.asarray(a.read().splitlines()).reshape(-1, 1)
X_test = list()
for i in range(0, X_test_lines.shape[0]):
    X_test.append(X_test_lines[i][0].split())

a = open(data_dir + "y_train", "r", encoding="utf-8")
y_train = a.read().splitlines()
for i in range(0, len(y_train)):
    y_train[i] = int(y_train[i])

a = open(data_dir + "y_test", "r", encoding="utf-8")
y_test = a.read().splitlines()
for i in range(0, len(y_test)):
    y_test[i] = int(y_test[i])

a.close()

print(X_train[1:20])  # debugging
print(X_test[1:20])  # debugging

# Feature engineering: One-hot encoding
choice = "bow"
if choice is "bow":
    X = np.concatenate((X_train, X_test)).reshape(-1, 1)

    print(f"x.shape: {X.shape}, x: {X[0:20]}")  # debugging

    cv = CountVectorizer()
    bag = cv.fit_transform(X)
elif choice is "enc":
    # combine dataset temporarily for one-hot encoding
    # X = np.concatenate((X_train, X_test))
    max_num_features = max(max(map(len, X_train)), max(map(len, X_test)))
    X = np.array([t + [None] * max_num_features for t in X_train])
    X = np.asarray(X).reshape(-1, 1)

    print(f"max: {max_num_features}")  # debugging
    print(f"shape: {X.shape}, x: {X[0:3]}")

    enc = OneHotEncoder(handle_unknown='ignore')
    enc_train_fit = enc.fit(X)
    X = enc_train_fit.transform(X).reshape(-1, 1)

    # split dataset again
    split = X_train.shape[0]
    X_train = X[0:split]
    X_test = X[split:X.shape[0]]  # rest
elif choice is "vectorize":
    pass  # TODO: vectorize; ML models need features, not just whole tweets; idea: cv(5,10)

# Shuffle data (keeps indices)
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

# Fitting the model
svm = SVC(kernel="linear", gamma="auto")
svm.fit(X_train, y_train)

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

""" DEBUG """
unique, counts = np.unique(svm.predict(X_test), return_counts=True)  # debug
print(f"X_test unique: {unique}, counts: {counts}, len: {len(svm.predict(X_test))}")  # debug
unique, counts = np.unique(y_test, return_counts=True)  # debug
print(f"y_test unique: {unique}, counts: {counts}")  # debug
