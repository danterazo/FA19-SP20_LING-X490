# LING-X 490: Abusive Language Detection
# Started FA19, finished SU20
# Dante Razo, drazo, 2020-05-15

from kaggle_preprocessing import read_data
from kaggle_build import build_main as build_datasets
from kaggle_build import export_df
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
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
        i = 1

        for set in sample[0]:  # for each set...
            data = pd.DataFrame(set)  # first member of tuple is the dataframe
            sample_type = sample[1]  # second member of tuple is a string
            print(f"===== {sample_type.capitalize()}-sample: pass {i} =====") if verbose else None

            # Store data as vectors
            X = data["comment_text"]  # initially reversed because it was easier to separate that way
            y = data["class"]

            # Model pipeline
            print("Instantiating model pipeline (CV & SVM)...") if verbose else None
            vec = CountVectorizer(analyzer="word", ngram_range=ngram_range)
            svc = SVC(C=1000, kernel="rbf", gamma=0.001)  # GridSearch best params
            clf = Pipeline([('vect', vec), ('svm', svc)])

            # Testing + results
            k = 5  # number of folds
            print(f"Fitting CountVectorizer & training {sample_type.capitalize()}-sample SVM...") if verbose else None

            scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
            scores_dict = cross_validate(clf, X, y, cv=k, n_jobs=14, scoring=scoring, return_train_score=True)
            scores_df = pd.DataFrame.from_dict(scores_dict)
            print("Training complete.\n")  # debugging, so is the one above. to remove

            # TODO: cross_val_predict + save `y_pred` to file, "\pred\pred.random1.csv"
            export_df(scores_df, sample_type, i, path="output/", filename="report")
            print(f"Report [{sample_type.lower()}, {analyzer}, ngram_range{ngram_range}]:\n {scores_df}\n")
            i += 1


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
# fit_data(rebuild, samples, analyzer, ngram_range, gridsearch, manual_boost, repeats, verbose, sample_size)
