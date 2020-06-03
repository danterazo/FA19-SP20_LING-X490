# LING-X 490
# This file builds, imports, and exports data
# Dante Razo, drazo
from kaggle_preprocessing import read_data, boost_data, sample_data
import pandas as pd
import os


# data headers: [y, X]

# only import once
def get_train():
    dataset = "train.target+comments.tsv"  # 'test' for classification problem
    return read_data(dataset)


# Gets 'n' posts, randomly selected, from the dataset. Then save to `.csv`
def build_random(data, sample_size, repeats=3):
    to_export = []
    # sample + export
    for i in range(0, repeats):
        to_export.append(sample_data(data, sample_size))

    export_data("random", to_export)


def build_boosted(data, manual_boost, sample_size, repeats=3):
    data_file = "train.target+comments.tsv"  # name for verbose prints
    to_export = []

    # sample + export, topic
    boosted_topic_data = boost_data(data, data_file, manual_boost)
    for i in range(0, repeats):
        to_export.append(sample_data(boosted_topic_data, sample_size))

    export_data("topic", to_export)

    # boost + sample + export, wordbank
    boosted_wordbank_data = boost_data(data, data_file)

    for i in range(0, repeats):
        to_export.append(sample_data(boosted_wordbank_data, sample_size))

    export_data("wordbank", to_export)


# save data to `.tsv`, `.csv`, etc.
def export_data(source, data, extension=".csv"):
    i = 1

    for d in data:
        filepath = os.path.join("../data/kaggle_data", f"train.{source}{i}{extension}")
        d.to_csv(filepath, index=False, header=False)
        i += 1


# generalized version of the above. `.csv`
def export_df(data, sample="no_sample", i="", path="", prefix="", index=True):
    filepath = os.path.join(path, f"{prefix}.{sample}{i}.csv")
    data.to_csv(filepath, index=index, header=True)


# builds one or both
def build_main(choice, topic, repeats, sample_size, verbose):
    """
    choice: choose which sample types to build. "random", "boosted", or "all"
    topic: topic for manual boosting
    """
    train = get_train()

    build_random(train, sample_size, repeats) if choice is "random" or "all" else None
    build_boosted(train, topic, sample_size, repeats) if choice is "boosted" or "all" else None
    print(f"Datasets built.") if verbose else None


# import Wiegand's lexicon, format it, and export it
# in `kaggle_build.py` because it isn't dynamic, i.e. the output is the same after every run
def build_lexicon():
    """ GOAL: create three-dimensional data
    1. Word
    2. Part of speech
    3. Class

    Then, manually remove non-abusive examples
    """
    data_dir = "../repos/lexicon-of-abusive-words/lexicons"  # common directory for all repos. assumes local sys
    dataset = "base"  # base | expanded

    names = ["word", "class"]
    data = pd.read_csv(f"{data_dir}/{dataset}Lexicon.txt", sep='\t', header=None, names=names)  # import Kaggle data

    split = [w.split("_") for w in data["word"]]  # split word and PoS

    data["part"] = [s[1] for s in split]  # remove PoS from words
    data["word"] = [s[0] for s in split]

    data.to_csv("lexicon_wiegand.csv", index=False)  # save to `.csv`

    # also export a lexicon of ONLY abusive words
    abusive = data[data["class"]]
    abusive["manual"] = ""
    abusive = abusive[["word", "class", "manual"]]

    filepath = os.path.join("../data/kaggle_data/lexicon", "lexicon.wiegand.abusive.csv")
    abusive.to_csv(filepath, index=False)  # save to `.csv`


""" MAIN """
# configuration
topic = ["trump"]  # [str]
to_build = "all"  # "all", "random", or "boosted"

# manually run:
# build_main(to_build, topic)
