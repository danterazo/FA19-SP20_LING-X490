# LING-X 490 Assignment 5: Waseem SVM
# Dante Razo, drazo, 10/30/2019
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sklearn.metrics
import pandas as pd
import random

# import datasets and create class vector
# ID, Diagnosis, mean features (x10), standard error features (x10), worst features (x10)
features = "ID Diagnosis meanRadius meanTexture meanPerimeter meanArea meanSmoothness meanCompactness meanConcavity " \
           "meanConcavePts meanSymmetry meanFractalDim seRadius seTexture sePerimeter seArea seSmoothness " \
           "seCompactness seConcavity seConcavePts seSymmetry seFractalDim worstRadius worstTexture worstPerimeter " \
           "worstArea worstSmoothness worstCompactness worstConcavity worstConcavePts worstSymmetry " \
           "worstFractalDim".split()
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                 names=features)
df = shuffle(df)  # shuffle data for k-folds cross-validation
classes = df.iloc[0:df.shape[0] + 1, 1]  # class vector from second column of data
classes = classes.replace(['B', 'M'], [0, 1])  # replace strings with ints for validation

df.drop('Diagnosis', axis=1, inplace=True)  # remove class column
df.drop('ID', axis=1, inplace=True)  # remove ID column (not needed)

# k-folds cross-validation with k=5
X_train, X_test, y_train, y_test = train_test_split(df, classes, test_size=0.8, train_size=0.2)

# Normalize data
df = sklearn.preprocessing.normalize(df, axis=0)

# Fitting the model
svm = SVC(kernel='rbf', gamma='scale')
svm.fit(X_train, y_train)

# Testing + results
print("Random/Baseline Accuracy:",
      sklearn.metrics.balanced_accuracy_score(y_test, [random.randint(0, 1) for x in range(1, (y_test.shape[0] + 1))]))
print("Testing Accuracy: ", sklearn.metrics.accuracy_score(y_test, svm.predict(X_test)))