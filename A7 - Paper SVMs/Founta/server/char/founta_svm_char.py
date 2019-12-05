# LING-X 490 Assignment 7: Founta SVM
# Dante Razo, drazo, 11/21/2019
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sklearn.metrics
import pandas as pd
import random


# Import data; TODO: remove httP://t.co/* links
def get_data():
    data_dir = "./data"

    hate = pd.read_csv(f"{data_dir}/hate.txt", sep='\n', names=["text"], engine='c')
    noHate = pd.read_csv(f"{data_dir}/noHate.txt", lineterminator='\n', names=["text"], engine='c')
    hate["class"] = 1  # abusive
    noHate["class"] = 0  # not abusive

    data = hate.append(noHate, ignore_index=False)
    X = data.iloc[:, 0]
    y = data["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test


# Feature engineering: vectorizer
# ML models need features, not just whole tweets
print("COUNTVECTORIZER CONFIG\n----------------------")
analyzer = "char"
ngram_upper_bound = [2, 3, 5, 10, 20]
kernel = "linear"

for i in ngram_upper_bound:
    X_train, X_test, y_train, y_test = get_data()
    verbose = True  # print statement flag

    vec = CountVectorizer(analyzer=analyzer, ngram_range=(1, int(i)))
    print("\nFitting CV...") if verbose else None
    X_train = vec.fit_transform(X_train.values.astype('U'))
    X_test = vec.transform(X_test.values.astype('U'))

    # Shuffle data (keeps indices)
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    # Fitting the model
    print("Training SVM...") if verbose else None
    svm = SVC(kernel=kernel, gamma="auto")  # tweak params
    svm.fit(X_train, y_train)
    print("Training complete.") if verbose else None

    # Testing + results
    rand_acc = sklearn.metrics.balanced_accuracy_score(y_test, [random.randint(0, 1) for x in range(0, len(y_test))])
    acc_score = sklearn.metrics.accuracy_score(y_test, svm.predict(X_test))

    print(f"\nResults for ({analyzer}, ngram_range(1,{i}):")
    print(f"Baseline Accuracy: {rand_acc}")  # random
    print(f"Testing Accuracy:  {acc_score}")
