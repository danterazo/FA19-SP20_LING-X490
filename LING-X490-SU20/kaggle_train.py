# LING-X 490: Abusive Language Detection
# Started FA19, finished SU20
# Dante Razo, drazo, 2020-05-15

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from kaggle_preprocessing import read_data
from kaggle_build import build_main as build_datasets


def fit_data(rebuild, samples, analyzer, ngram_range, gridsearch, manual_boost, repeats, verbose, sample_size):
    """
    verbose (boolean):  toggles print statements
    sample_size (int):  size of sampled datasets. If set too high, the smaller size will be used
    samples ([str]):    three modes: "random", "topic", "wordbank", or "all"
    analyzer (str):     either "word" or "char". for CountVectorizer
    ngram_range (()):   tuple containing lower and upper ngram bounds for CountVectorizer
    gridsearch (bool):  toggles SVM gridsearch functionality (significantly increases fit time)
    dev (bool):         toggles whether `dev` sets are used. False for `test` sets
    manual_boost ([str]):   use given array of strings for filtering instead of built-in wordbanks. Or pass `None`
    """

    build_datasets(samples, manual_boost, repeats, sample_size, verbose) if rebuild else None  # rebuild datasets

    # create list of lists: [[random1, random2, random3], [topic1, topic2, topic3], [wordbank1, wordbank2, wordbank3]]
    all_data = []
    for x in ["random", "topic", "wordbank"]:
        all_data.append((import_data(x), x))

    # choose one or the other sample type if desired
    if samples is "random":
        all_data = all_data[0]
    elif samples is "boosted":
        all_data = all_data[1:2]

    for sample in all_data:  # for each sample type...
        for i in range(1, repeats + 1):  # for each test...
            data = sample[0][i]  # first member of tuple is the dataframe
            sample_type = sample[1]  # second member of tuple is a string

            # X_train, X_test, y_train, y_test = data
            X = data.loc["comment_text"]  # initially reversed because it was easier to split that way
            y = data.loc["class"]

            # 5-Fold cross validation
            kf = KFold(n_splits=5, shuffle=False)
            fold_num = 1  # k-fold increment

            for train_index, test_index in kf.split(data):
                print(f"{sample_type.capitalize()}-sample pass {i}, fold {fold_num}") if verbose else None
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # debugging
                print(f"y_train: {y_train}")

                # Feature engineering w/ Vectorizer. ML models need features, not just whole tweets
                vec = CountVectorizer(analyzer="word", ngram_range=ngram_range)
                print(f"Fitting {sample_type.capitalize()}-sample CV...") if verbose else None
                X_train_CV = vec.fit_transform(X_train)
                X_test_CV = vec.transform(X_test)

                # Fitting the model
                print(f"Training {sample_type.capitalize()}-sample SVM...") if verbose else None

                if gridsearch:
                    svm_model = SVC()
                    svm_params = {'C': [0.1, 1, 10, 100, 1000],  # regularization param
                                  'gamma': ["scale", "auto", 1, 0.1, 0.01, 0.001, 0.0001],
                                  'kernel': ["linear", "poly", "rbf", "sigmoid"]}
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
                j += 1


""" IMPORT DATA """


def import_data(sample_type):
    to_return = []

    for i in range(1, 4):
        to_return.append(read_data(f"train.{sample_type}{i}.csv", delimiter="comma", verbose=False))

    return to_return


""" SCRIPT CONFIG """
samples = "all"  # "random", "boosted_topic", "boosted_wordbank", or "all"
analyzer = "word"  # "char" or "word"
ngram_range = (1, 3)  # int 2-tuple / couple
gridsearch = False  # bool. Leave 'FALSE'; best params hardcoded
dev = False  # bool
manual_boost = ["trump"]  # ["trump"]  # None, or an array of strings
rebuild = False  # rebuild datasets + export
repeats = 3  # number of datasets per sample type
verbose = True  # suppresses prints if FALSE
sample_size = 20000

""" MAIN """
fit_data(rebuild, samples, analyzer, ngram_range, gridsearch, manual_boost, repeats, verbose, sample_size)
