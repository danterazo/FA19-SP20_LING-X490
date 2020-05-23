# LING-X 490: Abusive Language Detection
# Started FA19, finished SU20
# Dante Razo, drazo, 2020-05-15

from kaggle_preprocessing import read_data
from kaggle_build import build_main as build_datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import pandas as pd


# for each fold of each dataset of each sample type, train an SVM
def fit_data(rebuild, samples, analyzer, ngram_range, gridsearch, manual_boost, repeats, verbose, sample_size):
    """
    rebuild (bool):     if TRUE, rebuild + rewrite the following datasets:
    samples ([str]):    three modes: "random", "boosted", or "all"
    analyzer (str):     either "word" or "char". for CountVectorizer
    ngram_range (()):   tuple containing lower and upper ngram bounds for CountVectorizer
    gridsearch (bool):  toggles SVM gridsearch functionality (significantly increases fit time)
    manual_boost ([str]):   use given array of strings for filtering instead of built-in wordbanks. Or pass `None`
    repeats (int):      controls the number of datasets built per sample type (if `rebuild` is TRUE)
    verbose (boolean):  toggles print statements
    sample_size (int):  size of sampled datasets. If set too high, the smaller size will be used
    """

    build_datasets(samples, manual_boost, repeats, sample_size, verbose) if rebuild else None  # rebuild datasets

    # struct example: [([random1, random2, ..., random_n], "random"), ...]
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
            data = pd.DataFrame(sample[0][i])  # first member of tuple is the dataframe
            sample_type = sample[1]  # second member of tuple is a string
            print(f"===== {sample_type.capitalize()}-sample: pass {i} =====") if verbose else None

            X = data["comment_text"]  # initially reversed because it was easier to separate that way
            y = data["class"]

            print("Instantiating model pipeline...") if verbose else None  # TODO: new progress
            vec = CountVectorizer(analyzer="word", ngram_range=ngram_range)
            svc = SVC(C=1000, kernel="rbf", gamma=0.001)  # GridSearch best params
            clf = Pipeline([('vect', vec), ('svm', svc)])
            # print("Pipeline (CountVec + SVM) instantiated.") if verbose else None  # TODO: new progress

            # get results
            k = 5  # number of folds
            print(f"Training {sample_type.capitalize()}-sample SVM...") if verbose else None
            print(cross_validate(clf, X, y, cv=k))
            print("Training complete.")  # debugging, so is the one above. to remove

            """
            print(f"Fitting {sample_type.capitalize()}-sample CV...") if verbose else None
            
            # 5-Fold cross validation
            kf = KFold(n_splits=5, shuffle=False)
            fold_num = 1  # k-fold increment for prints

            #print(f"===== {sample_type.capitalize()}-sample: pass {i}, fold {fold_num} =====") if verbose else None
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

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
            else:  # best params as determined by GridSearch
                svm_model = SVC(C=1000, kernel="rbf", gamma=0.001)  # GridSearch best params
                svm_fitted = svm_model.fit(X_train_CV, y_train)

            print(f"Training complete.") if verbose else None

            # Testing + results
            fold_num += 1
            print(f"\nClassification Report [{sample_type.lower()}, {analyzer}, ngram_range{ngram_range}]:\n "
                  f"{classification_report(y_test, svm_fitted.predict(X_test_CV), digits=6)}")
            """


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
manual_boost = ["trump"]  # ["trump"]  # None, or an array of strings
rebuild = False  # rebuild datasets + export
repeats = 3  # number of datasets per sample type
verbose = True  # suppresses prints if FALSE
sample_size = 20000

""" MAIN """
fit_data(rebuild, samples, analyzer, ngram_range, gridsearch, manual_boost, repeats, verbose, sample_size)
