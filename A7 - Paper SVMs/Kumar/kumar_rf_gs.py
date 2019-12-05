# LING-X 490 Assignment 7: Kumar Random Forest (w/ GridSearchCV)
# Dante Razo, drazo, 12/01/2019
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import GridSearchCV
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

for i in ngram_upper_bound:
    X_train, X_test, y_train, y_test = get_data()
    verbose = True  # print statement flag

    vec = CountVectorizer(analyzer=analyzer, ngram_range=(1, int(i)))
    print("\nFitting CV........") if verbose else None
    X_train = vec.fit_transform(X_train)
    X_test = vec.transform(X_test)

    # Shuffle data (keeps indices)
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    # RF parameter tuning w/ GridSearch
    rf_model = rf(n_jobs=1)
    rf_params = {'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200],
                 'criterion': ['gini', 'entropy']}
    rf_gs = GridSearchCV(rf_model, rf_params, n_jobs=4, cv=5)

    # Fitting the model
    print("Training RF/GS....") if verbose else None
    rf_gs.fit(X_train, y_train)
    print("Training complete.") if verbose else None

    # Testing + results
    rand_acc = sklearn.metrics.balanced_accuracy_score(y_test, [random.randint(0, 1) for x in range(0, len(y_test))])
    acc_score = sklearn.metrics.accuracy_score(y_test, rf_gs.predict(X_test))

    print(f"\nResults for ({analyzer}, ngram_range(1,{i}):")
    print(f"Baseline Accuracy: {rand_acc}")  # random
    print(f"Testing Accuracy:  {acc_score}")

""" RESULTS & DOCUMENTATION
# TUNING 

# CountVectorizer PARAM TESTING (GS) ; TODO
word, ngram_range(1,2):  0.7155555555555555
word, ngram_range(1,3):  0.7076767676767677
word, ngram_range(1,5):  0.693939393939394
word, ngram_range(1,10): 0.6953535353535354
word, ngram_range(1,20): 0.6866666666666666
char, ngram_range(1,2):  0.6880808080808081
char, ngram_range(1,3):  0.7058585858585859
char, ngram_range(1,5):  0.7141414141414142
char, ngram_range(1,10): 0.7262626262626263 *
char, ngram_range(1,20): 0.7024242424242424
"""
