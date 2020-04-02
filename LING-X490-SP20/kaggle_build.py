# LING-X490 SP20: Kaggle SVM
# Dante Razo, drazo
import pandas as pd


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

    print(f"data: {data.head}")

    # data.to_csv("lexicon_just-abusive.csv", index=False)  # save to `.csv`


# Takes Kaggle dataset, filters on topic, then saves new data to `.csv`
def filter_kaggle():
    pass


parse_lexicon()
