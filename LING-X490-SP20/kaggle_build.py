# LING-X490 SP20: Kaggle SVM
# Dante Razo, drazo
import pandas as pd
from sklearn.utils import shuffle


def parse_lexicon():
    """ GOAL: create three-dimensional data
    1. word
    2. Part of speech
    3. Class

    Then, manually remove non-abusive examples
    """
    data_dir = "../repos/lexicon-of-abusive-words/lexicons"  # common directory for all repos
    dataset = "base"  # base | expanded

    names = ["word", "class"]
    data = pd.read_csv(f"{data_dir}/{dataset}Lexicon.txt", sep='\t', header=None, names=names)  # import Kaggle data

    split = [w.split("_") for w in data["word"]]  # split word and PoS

    data["part"] = [s[1] for s in split]  # remove PoS from words
    data["word"] = [s[0] for s in split]

    return data


# only get abusive words
def get_abusive():
    data = parse_lexicon()

    abusive = data[data["class"]]
    abusive["manual"] = ""
    abusive = abusive[["word", "class", "manual"]]

    abusive.to_csv("lexicon.wiegand.abusive.csv", index=False)  # save to `.csv`
    return abusive


# read CSV, convert manual tags to: {true, false, NA}
def convert_manual_abusive():
    pass


# get stats on manually-classified dataset
def manual_analysis():
    pass


# Takes Kaggle dataset, filters on topic, then saves new data to `.csv`
def filter_kaggle():
    pass


# Gets 'n' posts, randomly selected, from the dataset
def random_kaggle(sample_size=10000):
    data_dir = "../data/kaggle_data"  # common directory for all datasets
    dataset = "train.target+comments.tsv"  # 'test' for classification problem
    print(f"Importing `{dataset}`...")  # progress indicator
    data_list = []  # temporary; used for constructing dataframe
    kaggle_threshold = 0.50  # for class vector

    # import data line-by-line
    with open(f"{data_dir}/{dataset}", "r", encoding="utf-8") as d:
        entries = d.readlines()

        for e in entries:
            splitLine = e.split("\t", 1)

            if len(splitLine) is 2:  # else: there's no score, so throw the example out
                data_list.append([float(splitLine[0]), splitLine[1]])

    # construct dataframe
    data = pd.DataFrame(data_list, columns=["class", "comment_text"])
    print(f"Data {data.shape} imported!")  # progress indicator

    # create class vector
    data.loc[data["class"] < kaggle_threshold, "class"] = 0
    data.loc[data["class"] >= kaggle_threshold, "class"] = 1

    # data processing
    data = shuffle(data)
    shuffled_data = data[0:sample_size]  # trim
    print(f"Data randomly sampled!")  # progress indicator

    # save data
    shuffled_data.to_csv("data_random-sample.csv", index=False)  # save to `.csv`
    print(f"Data saved!")  # progress indicator
    pass


""" MAIN """
# get_abusive()
# random_kaggle()
