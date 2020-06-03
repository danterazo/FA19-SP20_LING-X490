# LING-X 490
# This standalone file takes built data and reformats / averages / analyzes it
# Dante Razo, drazo
from kaggle_build import export_df, get_train
from kaggle_preprocessing import boost_data
import pandas as pd
import numpy as np
import os
import glob


def compare_lexicons():
    path = "../data/kaggle_data/lexicon_manual/"

    os.chdir(path)
    files = glob.glob('*.{}'.format("csv")) + glob.glob('*.{}'.format("tsv"))
    dfs = []

    # assumes they're all the same length (551, as was the provided lexicon)
    for filename in files:
        author = filename.split(".")[-2].strip()

        if author == "dante":
            dfs.append(lexicon_dante(filename))
        elif author == "dd":
            dfs.append(lexicon_dd(filename))
        elif author == "schaede":  # schaede
            dfs.append(lexicon_schaede(filename))

    df = pd.concat(dfs, axis=1)  # one big dataframe
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate "word" columns
    df["avg"] = df[df.columns[1:]].mean(axis=1)  # average class columns
    df["var"] = df[df.columns[1:]].var(axis=1)  # variance of class columns

    df["class"] = False
    df.loc[df.avg > 0.6, "class"] = True  # i.e. at least 2 people say its mildly
    export_df(df, sample="all", prefix="lexicon.manual")  # export it too


# calculate % examples in given data that contains abusive words
def percent_abusive(data, lex):
    """
    data (df): dataframe to filter
    lex (str): lexicon to filter with. Either "we" (wiegand extended) or "rds" (our manually tagged dataset)
    """

    filename = ""
    boost_list = []
    if lex == "rds":  # Razo, DD, Shaede
        filename = "../data/kaggle_data/lexicon_manual/lexicon.manual.all.csv"
        lexicon_rds = pd.read_csv(filename, sep=",", header=0)
        lexicon_rds = lexicon_rds[lexicon_rds["class"] == 1]  # only use abusive words (class=1)
        boost_list = list(lexicon_rds["word"])
    elif lex == "we":  # Wiegand
        filename = "../data/kaggle_data/lexicon/lexicon.wiegand.abusive.csv"
        lexicon_wiegand = pd.read_csv(filename, sep=",", header=0)
        boost_list = list(lexicon_wiegand["word"])  # todo: actually use expanded lexion

    # boost
    boosted_df = boost_data(data, data_file=filename, verbose=False, manual_boost=boost_list)
    return len(boosted_df) / len(data) * 100


""" CLEAN IMPORTED DATA """


# csv with extra columns
def lexicon_dante(filename):
    df = pd.read_csv(filename)[["word", "pass2"]]
    df.columns = ["word", "dante"]
    return df


# ssv
def lexicon_dd(filename):
    df = pd.read_csv(filename, sep='\t', header=0)[["word", "opinion"]]

    class_vec = []
    for x in df["opinion"]:
        manual_class = str(x).lower()

        if manual_class == "very abusive":
            class_vec.append(2)
        elif manual_class == "mildly abusive":
            class_vec.append(1)
        elif manual_class == "not abusive":
            class_vec.append(0)
        else:
            class_vec.append(np.NaN)

    df["dd"] = class_vec
    df = df[["word", "dd"]]
    return df


# csv
def lexicon_schaede(filename):
    df = pd.read_csv(filename, header=0).iloc[:, 0:2]
    df.columns = ["word", "opinion"]

    class_vec = []
    for x in df["opinion"]:
        manual_class = str(x).lower()

        if manual_class == "very abusive":
            class_vec.append(2)
        elif manual_class == "mildly abusive":
            class_vec.append(1)
        elif manual_class == "not abusive":
            class_vec.append(0)
        else:
            class_vec.append(np.NaN)

    df["schaede"] = class_vec
    df = df[["word", "schaede"]]
    return df


""" MAIN """
# compare_lexicons()
