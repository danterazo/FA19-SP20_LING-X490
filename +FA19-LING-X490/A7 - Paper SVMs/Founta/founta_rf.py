# LING-X 490 Assignment 7: Founta Random Forest
# Dante Razo, drazo, 11/21/2019
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sklearn.metrics
import pandas as pd
import random


# Import data; TODO: remove httP://t.co/* links
def get_data():
    data_dir = "data"

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
    X_train = vec.fit_transform(X_train.values.astype('U'))
    X_test = vec.transform(X_test.values.astype('U'))

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
# Criterion TESTING (n_estimators=100; analyzer=word, ngram_range(1,3)) ; TODO
Gini:    
Entropy: 

# CountVectorizer PARAM TESTING (n_estimators=100, criterion="gini", max_depth=2)
word, ngram_range(1,2):  0.9291376912378303
word, ngram_range(1,3):  0.9207927677329625
word, ngram_range(1,5):  0.9114742698191933
word, ngram_range(1,10): 0.9047983310152990
word, ngram_range(1,20): 0.9009040333796940
char, ngram_range(1,2):  0.9085535465924895
char, ngram_range(1,3):  0.9234353268428372
char, ngram_range(1,5):  0.9363004172461753
char, ngram_range(1,10): 0.9348400556328234
char, ngram_range(1,20): 0.9265646731571627
"""
