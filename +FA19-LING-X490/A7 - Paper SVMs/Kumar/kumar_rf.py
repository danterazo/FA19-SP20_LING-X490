# LING-X 490 Assignment 7: Kumar Random Forest
# Dante Razo, drazo, 11/21/2019
from sklearn.ensemble import RandomForestClassifier
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
    data_dir = "data"

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
print("PARAM CONFIG\n------------")
analyzer = input("CV: Please enter analyzer: ")  # CV param
ngram_upper_bound = input("CV: Please enter ngram upper bound(s): ").split()  # CV param
n_estimators = input("RF: Please enter # of estimators: ")  # RF param
criterion = input("RF: Please enter criterion: ")  # RF param; gini OR entropy

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
    print("Training RF...") if verbose else None
    rf = RandomForestClassifier(n_estimators=int(n_estimators), criterion=criterion)
    rf.fit(X_train, y_train)
    print("Training complete.") if verbose else None

    # Testing + results
    rand_acc = sklearn.metrics.balanced_accuracy_score(y_test, [random.randint(0, 1) for x in range(0, len(y_test))])
    acc_score = sklearn.metrics.accuracy_score(y_test, rf.predict(X_test))

    print(f"\nResults for ({analyzer}, ngram_range(1,{i}):")
    print(f"Baseline Accuracy: {rand_acc}")  # random
    print(f"Testing Accuracy:  {acc_score}")

""" RESULTS & DOCUMENTATION
# Criterion TESTING (n_estimators=100; analyzer=word, ngram_range(1,3))
Gini:    0.6876767676767677
Entropy: 0.7101010101010101

# CountVectorizer PARAM TESTING (n_estimators=100, criterion="gini", max_depth=2)
word, ngram_range(1,2):  0.706060606060606
word, ngram_range(1,3):  0.6862626262626262
word, ngram_range(1,5):  0.6872727272727273
word, ngram_range(1,10): 0.675959595959596
word, ngram_range(1,20): 0.6745454545454546
char, ngram_range(1,2):  0.6795959595959596
char, ngram_range(1,3):  0.6933333333333334
char, ngram_range(1,5):  0.7107070707070707 *
char, ngram_range(1,10): 0.6937373737373738
char, ngram_range(1,20): 0.6890909090909091
"""
