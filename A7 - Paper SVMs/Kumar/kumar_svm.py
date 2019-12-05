# LING-X 490 Assignment 7: Kumar SVM
# Dante Razo, drazo, 11/21/2019
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sklearn.metrics
import pandas as pd
import random


def get_data():
    """ KUMAR Dataset Documentation
    - Hate:
        - Overtly Aggressive (OAG)
        - Covertly Aggressive (CAG)
    - Not Hate:
        - Non-aggressive (NAG)
    """
    data_dir = "./data"

    # combine data
    cag = pd.read_csv(f"{data_dir}/cag.txt", sep='\n', names=["text"])
    oag = pd.read_csv(f"{data_dir}/oag.txt", sep='\n', names=["text"])
    nag = pd.read_csv(f"{data_dir}/nag.txt", sep='\n', names=["text"])
    cag["class"] = 1  # 1: abusive (Kumar parlance: "aggresive")
    oag["class"] = 1
    nag["class"] = 0  # 0: not abusive

    # combine then split into X and y
    data = cag.append(oag, ignore_index=True).append(nag, ignore_index=True)
    X = data.iloc[:, 0]
    y = data["class"]

    # split into train, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    return X_train, X_test, y_train, y_test


# Feature engineering: vectorizer
# ML models need features, not just whole tweets
print("COUNTVECTORIZER CONFIG\n----------------------")
analyzer = input("Please enter CV analyzer: ")  # CV param
ngram_upper_bound = input("Please enter CV ngram upper bound(s): ").split()  # CV param
kernel = input("Please enter SVM kernel: ")  # SVM param

for i in ngram_upper_bound:
    X_train, X_test, y_train, y_test = get_data()
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
    svm = SVC(kernel=kernel, gamma="auto")  # tweak params
    svm.fit(X_train, y_train)
    print("Training complete.") if verbose else None

    # Testing + results
    rand_acc = sklearn.metrics.balanced_accuracy_score(y_test, [random.randint(0, 1) for x in range(0, len(y_test))])
    acc_score = sklearn.metrics.accuracy_score(y_test, svm.predict(X_test))

    print(f"\nResults for ({analyzer}, ngram_range(1,{i}):")
    print(f"Baseline Accuracy: {rand_acc}")  # random
    print(f"Testing Accuracy:  {acc_score}")

""" RESULTS & DOCUMENTATION
# KERNEL TESTING (gamma="auto"; analyzer=word, ngram_range(1,3))
linear:  0.6987878787878787
rbf:     0.5812121212121212
poly:    0.5892929292929293
sigmoid: 0.5793939393939394
precomputed: N/A, not supported

# CountVectorizer PARAM TESTING (kernel="linear") ; TODO
word, ngram_range(1,2):  0.7030303030303030 *
word, ngram_range(1,3):  0.7010101010101010 # independent from linear kernel test
word, ngram_range(1,5):  0.6890909090909091
word, ngram_range(1,10): 0.6826262626262626
word, ngram_range(1,20): 0.6890909090909091
char, ngram_range(1,2):  0.6709090909090909
char, ngram_range(1,3):  0.6652525252525252
char, ngram_range(1,5):  0.6761616161616162
char, ngram_range(1,10): 0.6919191919191919
char, ngram_range(1,20): 0.6636363636363637
"""
