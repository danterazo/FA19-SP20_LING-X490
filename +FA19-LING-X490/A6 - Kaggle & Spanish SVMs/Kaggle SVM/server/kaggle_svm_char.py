# LING-X 490 Assignment 6: Kaggle SVM
# Dante Razo, drazo, 11/14/2019
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics
import pandas as pd
import random


# Import data; TODO: remove httP://t.co/* links
# original Kaggle dataset: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
def get_data():
    # TODO: remove NEWLINE_TOKEN; .replace("NEWLINE_TOKEN", "")
    data_dir = "../data"

    data = pd.read_csv(f"{data_dir}/toxicity_annotated_comments.tsv", sep='\t', header=0)
    train = data.loc[data['split'] == "train"]
    test = data.loc[data['split'] == "test"]
    dev = data.loc[data['split'] == "dev"]

    X_train = train.iloc[:, 1]
    X_test = test.iloc[:, 1]
    X_dev = dev.iloc[:, 1]  # what to do with this? validate?

    y = 3  # assumes that 'logged_in' is the class feature
    y_train = train.iloc[:, y] * 1
    y_test = test.iloc[:, y] * 1
    y_dev = dev.iloc[:, y] * 1

    return X_train, X_test, X_dev, y_train, y_test, y_dev


# Feature engineering: vectorizer
# ML models need features, not just whole tweets
print("COUNTVECTORIZER CONFIG\n----------------------")
# analyzer = input("Please enter analyzer: ")
# ngram_upper_bound = input("Please enter ngram upper bound(s): ").split()
analyzer = "char"
ngram_upper_bound = [2, 3, 5, 10, 20]

for i in ngram_upper_bound:
    X_train, X_test, X_dev, y_train, y_test, y_dev = get_data()
    verbose = True  # print statement flag

    vec = CountVectorizer(analyzer=analyzer, ngram_range=(1, int(i)))
    print("\nFitting CV...") if verbose else None
    X_train = vec.fit_transform(X_train)
    X_test = vec.transform(X_test)

    # Shuffle data (keeps indices)
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    # Fitting the model
    print("Training SVM...") if verbose else None
    svm = SVC(kernel="linear", gamma="auto")  # TODO: tweak params
    svm.fit(X_train, y_train)
    print("Training complete.") if verbose else None

    # Testing + results
    rand_acc = sklearn.metrics.balanced_accuracy_score(y_test, [random.randint(0, 1) for x in range(0, len(y_test))])
    acc_score = sklearn.metrics.accuracy_score(y_test, svm.predict(X_test))

    print(f"\nResults for ({analyzer}, ngram_range(1,{i}):")
    print(f"Baseline Accuracy: {rand_acc}")  # random
    print(f"Testing Accuracy:  {acc_score}")
