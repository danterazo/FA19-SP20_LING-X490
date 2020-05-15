# LING-X 490: Abusive Language Detection
# Started FA19, finished SU20
# Dante Razo, drazo, 2020-05-15

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from string import capwords
import sklearn.metrics
import pandas as pd

""" GLOBAL VARIABLES """
# pd.options.mode.chained_assignment = None  # suppress SettingWithCopyWarning
verbose = True  # if I didn't define it globally then I'd be passing it to every f() like a React prop


def get_data(dev):
    random_data = get_random_data()
    boosted_data = get_boosted_data()

    # trim both to size if necessary
    if len(random_data) is not len(boosted_data):
        min_sample_size = min(len(random_data), len(boosted_data))

        random_data = random_data[0:min_sample_size]  # trim
        boosted_data = boosted_data[0:min_sample_size]  # trim

    # split data into X, y
    random_splits = split_data(random_data, dev)
    boosted_splits = split_data(boosted_data, dev)

    # return data and identifiers
    return [[random_splits, "random"], [boosted_splits, "boosted"]]


# split into: train, test, dev
def split_data(data, dev, shuffle=True):
    X = data.loc[:, data.columns != "class"]
    y = data.loc[:, data.columns == "class"]

    # train: 60%, dev: 20%, test: 20%
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                test_size=0.2,
                                                                                shuffle=shuffle,
                                                                                random_state=42)

    X_train, X_dev, y_train, y_dev = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                              test_size=0.25,
                                                                              shuffle=shuffle,
                                                                              random_state=42)

    # use dev sets if dev=TRUE
    return [X_train, X_dev, y_train, y_dev] if dev else [X_train, X_test, y_train, y_test]


def read_data(dataset, delimiter, verbose=True):
    data_dir = "../data/kaggle_data"  # common directory for all datasets
    print(f"Importing `{dataset}`...") if verbose else None  # progress indicator
    data_list = []  # temporary; used for constructing dataframe

    # import data
    with open(f"{data_dir}/{dataset}", "r", encoding="utf-8") as d:
        entries = d.readlines()

        for e in entries:
            if delimiter is "tab":
                splitLine = e.split("\t", 1)
            else:
                splitLine = e.split(",", 1)

            if len(splitLine) is 2:  # else: there's no score, so throw the example out
                data_list.append([float(splitLine[0]), splitLine[1]])

    data = pd.DataFrame(data_list, columns=["score", "comment_text"])
    print(f"Data {data.shape} imported!") if verbose else None  # progress indicator

    kaggle_threshold = 0.50  # from Kaggle documentation (see page)
    dev = True  # set to FALSE when its time to validate `train` dataset
    shuffle = True  # self-explanatory
    to_return = []  # this function returns a list of lists. Each inner list contains `X` and `y`

    # create class vector
    data["class"] = 0
    data.loc[data.score >= kaggle_threshold, "class"] = 1

    return data


# already defined. just import
def get_random_data():
    return read_data("train.random.csv", delimiter="comma")


# TODO
def get_boosted_data():
    df = read_data("train.target+comments.tsv", delimiter="tab")

    # WIP
    return df


""" WORK """


def main(verbose, sample_size, samples, analyzer, ngram_range, gridsearch):
    """
    verbose (boolean):  toggles print statements
    sample_size (int):  size of sampled datasets. If set too high, the smaller size will be used
    samples ([str]):    three choices: "boosted" for only boosted, "random", or "both"
    analyzer (str):     either "word" or "char". for CountVectorizer
    ngram_range (()):   tuple containing lower and upper ngram bounds for CountVectorizer
    gridsearch (bool):  toggles SVM gridsearch functionality (significantly increases fit time)
    dev (bool):         toggles whether `dev` sets are used. False for `test` sets
    """

    all_data = get_data(verbose, sample_size)  # array of data. [[boosted X,y], [random X,y]]

    # choose one or the other if applicable
    if samples is "random":
        all_data = all_data[0]
    elif samples is "boosted":
        all_data = all_data[1]

    for sample in all_data:
        data = sample[0]  # first member of tuple is an array of splits
        sample_type = sample[1]  # second member of tuple is a string

        X_train, X_test, y_train, y_test = data

        # Feature engineering: Vectorizer. ML models need features, not just whole tweets
        vec = CountVectorizer(analyzer="word", ngram_range=ngram_range)
        print(f"\nFitting {sample_type.capitalize()}-sample CV...") if verbose else None
        X_train_CV = vec.fit_transform(X_train["comment_text"])
        X_test_CV = vec.transform(X_test["comment_text"])

        # Fitting the model
        print(f"Training {sample_type.capitalize()}-sample SVM...") if verbose else None
        svm_model = SVC(kernel="linear")

        if gridsearch:
            svm_params = {'C': [0.1, 1, 10, 100, 1000],  # regularization param
                          'gamma': ["scale", "auto", 1, 0.1, 0.01, 0.001, 0.0001],  # kernel coefficient (R, P, S)
                          'kernel': ["linear", "poly", "rbf", "sigmoid"]}  # SVM kernel (precomputed not supported)
            svm_gs = GridSearchCV(svm_model, svm_params, n_jobs=4, cv=5)
            svm_fitted = svm_gs.fit(X_train_CV, y_train.values.ravel())
        else:
            svm_fitted = svm_model.fit(X_train, y_train)

        print(f"Training complete.") if verbose else None

        # Testing + results
        print(f"\nClassification Report [{sample_type.lower()}, {analyzer}, ngram_range(1,{i})]:\n "
              f"{classification_report(y_test, svm_fitted.predict(X_test_CV), digits=6)}")


""" MAIN """
sample_size = 10000  # formerly 15000
samples = "Both"
analyzer = "word"
ngram_range = (1, 1)
gridsearch = True
dev = False

main(verbose, sample_size, samples, analyzer, ngram_range, gridsearch)
