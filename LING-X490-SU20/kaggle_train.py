# LING-X 490: Abusive Language Detection
# Started FA19, finished SU20
# Dante Razo, drazo, 2020-05-15

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import pandas as pd


from kaggle_export import *
from kaggle_preprocessing import *

""" SCRIPT CONFIG """
verbose = True  # if I didn't define it globally then I'd be passing it to every f() like a React prop
system = "local"  # if defined as "server", will change relative paths for dept NLP server
sample_size = 20000  # int
samples = "both"  # "random", "boosted_topic", "boosted_wordbank", or "all"
analyzer = "word"  # "char" or "word"
ngram_range = (1, 3)  # int 2-tuple / couple
gridsearch = False  # bool. Leave 'FALSE'; best params hardcoded
dev = False  # bool
manual_boost = ["trump"]  # ["trump"]  # None, or an array of strings

fit_data(verbose, sample_size, samples, analyzer, ngram_range, gridsearch, manual_boost)


def fit_data(verbose, sample_size, samples, analyzer, ngram_range, gridsearch, manual_boost):
    """
    verbose (boolean):  toggles print statements
    sample_size (int):  size of sampled datasets. If set too high, the smaller size will be used
    samples ([str]):    three modes: "boosted_topic", "boosted_wordbank", "random", or "all"
    analyzer (str):     either "word" or "char". for CountVectorizer
    ngram_range (()):   tuple containing lower and upper ngram bounds for CountVectorizer
    gridsearch (bool):  toggles SVM gridsearch functionality (significantly increases fit time)
    dev (bool):         toggles whether `dev` sets are used. False for `test` sets
    manual_boost ([str]):   use given array of strings for filtering instead of built-in wordbanks. Or pass `None`
    """

    # array of data. [[random X,y], [boosted_topic X,y], [boosted_wordbank X,y]]
    # TODO: don't import others if only one is requested
    all_data = get_data(verbose, sample_size, manual_boost)

    # choose one or the other if applicable
    if samples is "random":
        all_data = all_data[0]
    elif samples is "boosted_topic":
        all_data = all_data[1]
    elif samples is "boosted_wordbank":
        all_data = all_data[2]

    for sample in all_data:
        data = sample[0]  # first member of tuple is an array of splits
        sample_type = sample[1]  # second member of tuple is a string

        X_train, X_test, y_train, y_test = data

        # Feature engineering: Vectorizer. ML models need features, not just whole tweets
        vec = CountVectorizer(analyzer="word", ngram_range=ngram_range)
        print(f"Fitting {sample_type.capitalize()}-sample CV...") if verbose else None
        X_train_CV = vec.fit_transform(X_train["comment_text"])
        X_test_CV = vec.transform(X_test["comment_text"])

        # Fitting the model
        print(f"Training {sample_type.capitalize()}-sample SVM...") if verbose else None

        if gridsearch:
            svm_model = SVC()
            svm_params = {'C': [0.1, 1, 10, 100, 1000],  # regularization param
                          'gamma': ["scale", "auto", 1, 0.1, 0.01, 0.001, 0.0001],  # kernel coefficient (R, P, S)
                          'kernel': ["linear", "poly", "rbf", "sigmoid"]}  # SVM kernel (precomputed not supported)
            svm_gs = GridSearchCV(svm_model, svm_params, n_jobs=12, cv=5)
            svm_fitted = svm_gs.fit(X_train_CV, y_train.values.ravel())
            print(f"GridSearchCV SVM Best Params: {svm_gs.best_params_}")
        else:
            # GridSearchCV SVM Best Params: {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
            svm_model = SVC(C=1000, kernel="rbf", gamma=0.001)  # GridSearch best params
            svm_fitted = svm_model.fit(X_train, y_train)

        print(f"Training complete.") if verbose else None

        # Testing + results
        print(f"\nClassification Report [{sample_type.lower()}, {analyzer}, ngram_range{ngram_range}]:\n "
              f"{classification_report(y_test, svm_fitted.predict(X_test_CV), digits=6)}")


""" IMPORT DATA """


def get_data(dev, sample_size, manual_boost):
    random_data = get_random_data()
    boosted_topic_data = get_boosted_data(manual_boost)
    boosted_wordbank_data = get_boosted_data()

    # split data into X, y
    # TODO: don't split; let 5CV do the work
    random_splits = split_data(random_data, dev)
    topic_splits = split_data(boosted_topic_data, dev)
    wordbank_splits = split_data(boosted_wordbank_data, dev)

    # return data and identifiers
    return [[random_splits, "random"], [topic_splits, "boosted (topic)"], [wordbank_splits, "boosted (wordbank)"]]


# already saved as `.csv`. just import
def get_random_data():
    # TODO: import all three datasets at once + average results
    return read_data("train.random1.csv", delimiter="comma")


def get_boosted_data(manual_boost=None):
    data_file = "train.target+comments.tsv"  # only imports dataset once
    data = read_data(data_file, delimiter="tab")

    boosted_data = boost_data(data, data_file, manual_boost)
    return boosted_data
