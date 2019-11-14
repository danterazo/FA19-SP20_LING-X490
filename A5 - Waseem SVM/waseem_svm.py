# LING-X 490 Assignment 5: Waseem SVM
# Dante Razo, drazo, 10/30/2019
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics
import numpy as np
import random

# Import data
# TODO: idea: remove http://t.co/* links
data_dir = "./data/"
a = open(data_dir + "X_train", "r", encoding="utf-8")
X_train = a.read().splitlines()

a = open(data_dir + "X_test", "r", encoding="utf-8")
X_test = a.read().splitlines()

a = open(data_dir + "y_train", "r", encoding="utf-8")
y_train = a.read().splitlines()
for i in range(0, len(y_train)):
    y_train[i] = int(y_train[i])

a = open(data_dir + "y_test", "r", encoding="utf-8")
y_test = a.read().splitlines()
for i in range(0, len(y_test)):
    y_test[i] = int(y_test[i])

a.close()

# Feature engineering: vectorizer
# ML models need features, not just whole tweets
print("COUNTVECTORIZER CONFIG\n----------------------")
analyzer = input("Please enter analyzer: ")
ngram_upper_bound = input("Please enter ngram_upper_bound: ")

vec = CountVectorizer(analyzer=analyzer, ngram_range=(1, int(ngram_upper_bound)))  # TODO: word vs char, ngram_range
print("\nFitting CV...")
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# Shuffle data (keeps indices)
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

# Fitting the model
print("Training SVM...")
svm = SVC(kernel="linear", gamma="auto")  # TODO: tweak params
svm.fit(X_train, y_train)
print("Training complete.\n")

""" KERNEL RESULTS (gamma="auto"), ngram_range(1,3)
linear: 0.8549618320610687
rbf: 0.6844783715012722
poly: 0.6844783715012722
sigmoid: 0.6844783715012722
precomputed: N/A, not supported
"""

# Testing + results
rand_acc = sklearn.metrics.balanced_accuracy_score(y_test, [random.randint(1, 2) for x in range(0, len(y_test))])
print(f"Random/Baseline Accuracy: {rand_acc}")
print(f"Testing Accuracy: {sklearn.metrics.accuracy_score(y_test, svm.predict(X_test))}")

""" CV PARAM TESTING (kernel="linear")
word, ngram_range(1,2):  0.8606870229007634
word, ngram_range(1,3):  0.8549618320610687
word, ngram_range(1,5):  0.8473282442748091
word, ngram_range(1,10): 0.8358778625954199
char, ngram_range(1,2):  
char, ngram_range(1,3):  
char, ngram_range(1,5):  
char, ngram_range(1,10): 
"""
