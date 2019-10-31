# LING-X 490 Assignment 5: Waseem SVM
# Dante Razo, drazo, 10/30/2019
from sklearn.svm import SVC
from sklearn import preprocessing
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

# Feature engineering: One-hot encoding or Label encoding
choice = "enc"
if choice is "enc":
    # combine dataset temporarily for one-hot encoding
    X = np.concatenate((X_train, X_test))

    enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
    enc_train_fit = enc.fit(X)
    X = enc_train_fit.transform(X)

    # split dataset again
    split = X_train.shape[0]
    X_train = X[0:split]
    X_test = X[split:X.shape[0]]  # rest

"""
# issues w/ number of features
if choice is "enc-old":
    enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
    enc_train_fit = enc.fit(X_train)
    X_train = enc_train_fit.transform(X_train)

    enc_test_fit = enc.fit(X_test)
    X_test = enc_test_fit.transform(X_test)

if choice is "le":
    le = preprocessing.LabelEncoder()
    le_train_fit = le.fit(X_train)
    X_train = le_train_fit.transform(X_train)

    le_test_fit = le.fit(X_test)
    X_test = le_test_fit.transform(X_test)
"""

# Fitting the model
svm = SVC(kernel='rbf', gamma='scale')
svm.fit(X_train, y_train)

# Testing + results
rand_acc = sklearn.metrics.balanced_accuracy_score(y_test, [random.randint(1, 2) for x in range(0, len(y_test))])
print(f"Random/Baseline Accuracy: {rand_acc}")
print(f"Testing Accuracy: {sklearn.metrics.accuracy_score(y_test, svm.predict(X_test))}")
