# LING-X 490
# This standalone file takes existing data and reformats / averages it
# Dante Razo, drazo
from kaggle_build import export_df
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
    export_df(df, sample="all", filename="lexicon.manual")


""" IMPORTS """


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
compare_lexicons()
