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
print("COUNTVECTORIZER CONFIG\n----------------------")
analyzer = input("Please enter CV analyzer: ")  # CV param
ngram_upper_bound = input("Please enter CV ngram upper bound(s): ").split()  # CV param

for i in ngram_upper_bound:
    X_train, X_test, y_train, y_test = get_data()
    verbose = True  # print statement flag

    vec = CountVectorizer(analyzer=analyzer, ngram_range=(1, int(i)))
    print("\nFitting CV........") if verbose else None
    X_train = vec.fit_transform(X_train.values.astype('U'))
    X_test = vec.transform(X_test.values.astype('U'))

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

# CountVectorizer PARAM TESTING (GS)
word, ngram_range(1,2):  0.9296244784422809
word, ngram_range(1,3):  0.9228789986091794
word, ngram_range(1,5):  0.9123783031988874
word, ngram_range(1,10): 0.9037552155771905
word, ngram_range(1,20): 0.9013908205841447
char, ngram_range(1,2):  0.9081363004172461
char, ngram_range(1,3):  0.9248261474269819
char, ngram_range(1,5):  0.9369262865090403
char, ngram_range(1,10): 0.9347705146036162
char, ngram_range(1,20): 0.9270514603616133
"""
