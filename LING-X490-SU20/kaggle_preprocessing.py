# LING-X490 SU20:
# This file imports and processes data for use in SVM
# Dante Razo, drazo
import sklearn.metrics
import pandas as pd
import os
import re


# read + process training data
def read_data(dataset, delimiter, verbose=True):
    data_dir = "../data/kaggle_data"  # common directory for all datasets

    print(f"Importing `{dataset}`...") if verbose else None  # progress indicator
    data_list = []  # temporary; used for constructing dataframe

    # import data
    with open(f"{data_dir}/{dataset}", "r", encoding="utf-8") as d:
        entries = d.readlines()

        for e in entries:
            if delimiter is "tab":  # tsv
                split_line = e.split("\t", 1)
            else:  # default: csv
                split_line = e.split(",", 1)

            if len(split_line) is 2:  # else: there's no score, so throw the example out
                data_list.append([float(split_line[0]), str(split_line[1])])

    data = pd.DataFrame(data_list, columns=["score", "comment_text"])
    print(f"Data {data.shape} imported!\n") if verbose else None  # progress indicator

    kaggle_threshold = 0.50  # from Kaggle documentation (see page)

    # create class vector
    data["class"] = 0
    data.loc[data.score >= kaggle_threshold, "class"] = 1

    # remove score vector; replaced by class (bool) vector
    data = data.drop(columns="score")

    # swap column/feature order
    data = data[["class", "comment_text"]]
    return data


# split into: train, test, dev
def split_data(data, dev, shuffle=False):
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


# boosts data based on given topics or predefined wordbanks
def boost_data(data, data_file, verbose, manual_boost=None):
    print(f"Boosting data...") if verbose else None

    boosted_data = filter_data(data, data_file, verbose, manual_boost)

    print(f"Data boosted!\n") if verbose else None
    return boosted_data


# shuffle data then sample from it
def sample_data(data, size):
    return data.sample(frac=1)[0:size]


# return data that contains any word in Wordbank
def filter_data(data, data_file, verbose, manual_boost=None):
    """
    data (df):          dataset to filter
    topics ([str]]):    word(s) to filter with. this wordbank bypasses the banks below
    """
    if verbose:
        if manual_boost:
            print(f"Filtering `{data_file}` on {manual_boost}...")
        else:
            print(f"Filtering `{data_file}` on wordbank...")

    # source (built upon): https://dictionary.cambridge.org/us/topics/religion/islam/d
    islam_wordbank = ["allah", "caliphate", "fatwa", "hadj", "hajj", "halal", "headscarf", "hegira", "hejira",
                      "hijab", "islam", "islamic", "jihad", "jihadi", "mecca", "minaret", "mohammeden", "mosque",
                      "muhammad", "mujahideen", "muslim", "prayer", "mat", "prophet", "purdah", "ramadan", "salaam",
                      "sehri", "sharia", "shia", "sunni", "shiism", "sufic", "sufism", "suhoor", "sunna", "koran",
                      "qur'an", "yashmak", "ISIS", "ISIL", "al-Qaeda", "Taliban"]

    # TODO: see Sandra's email for suggestions

    # source: https://www.usatoday.com/story/news/2017/03/16/feminism-glossary-lexicon-language/99120600/
    metoo_wordbank = ["metoo", "feminism", "victim", "consent", "patriarchy", "sexism", "misogyny", "misandry",
                      "misogynoir", "cisgender", "transgender", "transphobia", "transmisogyny", "terf", "swef",
                      "non-binary", "woc", "victim-blaming", "trigger", "privilege", "mansplain", "mansplaining",
                      "manspread", "manspreading", "woke", "feminazi"]

    # source: https://en.wikipedia.org/wiki/Wikipedia:List_of_controversial_issues#Politics_and_economics
    politics_wordbank = ["republican", "GOP", "democrats", "liberal", "liberals", "abortion", "brexit",
                         "anti-semitism", "atheism", "conservatives", "capitalism", "communism", "Cuba", "fascism",
                         "Fox News", "immigration", "kashmir", "harambe", "israel", "hitler", "mexico",
                         "neoconservatism", "neoliberalism", "palestine", "9/11", "socialism", "Clinton", "Trump",
                         "Sanders", "Guantanamo", "torture", "Flight 77", "Marijuana", "sandinistas"]

    # source: https://en.wikipedia.org/wiki/Wikipedia:List_of_controversial_issues#History
    history_wordbank = ["Apartheid", "Nazi", "Black Panthers", "Rwandan Genocide", "Jim Crow", "Ku Klux Klan"]

    # source: https://en.wikipedia.org/wiki/Wikipedia:List_of_controversial_issues#Religion
    religion_wordbank = ["jew", "judaism", "christian", "christianity", "Jesus Christ", "Baptist", "WASP",
                         "Protestant", "Westboro Baptist Church"]

    # source: Sandra's suggestions, email from 2020-03-16
    sandra_wordbank = ["trump", "obama", "trudeau", "clinton", "hillary", "donald", "tax", "taxpayer", "vote",
                       "voting", "election", "party", "president", "politician", "women", "woman", "fact",
                       "military", "citizen", "nation", "church", "christian", "muslim", "liberal", "democrat",
                       "republican", "religion", "religious", "administration", "immigrant", "gun", "science",
                       "freedom", "solution", "corporate"]

    # words with special capitalization rules; except from capwords() function call below
    special_caps = ["al-Qaeda", "CNN", "KKK", "LGBT", "LGBTQ", "LGBTQIA"]

    # manually observed abusive words in explicit examples
    explicitly_abusive = ["sh*tty"]

    # future, TODO: https://thebestschools.org/magazine/controversial-topics-research-starter/

    if not manual_boost:
        # combine the above wordbanks
        combined_topics = islam_wordbank + metoo_wordbank + politics_wordbank + history_wordbank + religion_wordbank + \
                          sandra_wordbank + special_caps + explicitly_abusive
    else:
        # use the given topics (arg)
        combined_topics = manual_boost

    topic = combined_topics  # easy toggle if you want to focus on a specific topic instead
    wordbank = list(dict.fromkeys(topic))  # remove dupes

    # wordbank = wordbank + ["#" + word for word in topic]  # ...then add hashtags for all words
    wordbank = list(dict.fromkeys(wordbank))  # remove dupes again cause once isn't enough for some reason
    wordbank_regex = re.compile("|".join(wordbank), re.IGNORECASE)  # compile regex. case insensitive

    # idea: .find() for count. useful for threshold
    filtered_data = data[data["comment_text"].str.contains(wordbank_regex)]
    print(f"Data filtered to size {filtered_data.shape[0]}.") if verbose else None
    return filtered_data


def parse_lexicon():
    """ GOAL: create three-dimensional data
    1. word
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

    filepath = os.path.join("../data/kaggle_data/lexicon", "lexicon_wiegand_just-abusive.csv")
    abusive.to_csv(filepath, index=False)  # save to `.csv`
