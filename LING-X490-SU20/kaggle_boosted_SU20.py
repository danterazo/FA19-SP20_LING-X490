# LING-X 490: Abusive Language Detection
# Started FA19, finished SU20
# Dante Razo, drazo, 2020-05-15

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import sklearn.metrics
import pandas as pd
import re

""" GLOBAL VARIABLES """
# pd.options.mode.chained_assignment = None  # suppress SettingWithCopyWarning
verbose = True  # if I didn't define it globally then I'd be passing it to every f() like a React prop


def fit_data(verbose, sample_size, samples, analyzer, ngram_range, gridsearch, manual_boost):
    """
    verbose (boolean):  toggles print statements
    sample_size (int):  size of sampled datasets. If set too high, the smaller size will be used
    samples ([str]):    three modes: "boosted_topic", "boosted_wordbank", "random", or "all"
    analyzer (str):     either "word" or "char". for CountVectorizer
    ngram_range (()):   tuple containing lower and upper ngram bounds for CountVectorizer
    gridsearch (bool):  toggles SVM gridsearch functionality (significantly increases fit time)
    dev (bool):         toggles whether `dev` sets are used. False for `test` sets
    manual_boost ([str]):   use given array of strings for filtering instead of built-in wordbanks. Or pass `None`
    """

    # array of data. [[random X,y], [boosted_topic X,y], [boosted_wordbank X,y]]
    # TODO: don't import others if only one is requested
    all_data = get_data(verbose, sample_size, manual_boost)

    # choose one or the other if applicable
    if samples is "random":
        all_data = all_data[0]
    elif samples is "boosted_topic":
        all_data = all_data[1]
    elif samples is "boosted_wordbank":
        all_data = all_data[2]

    for sample in all_data:
        data = sample[0]  # first member of tuple is an array of splits
        sample_type = sample[1]  # second member of tuple is a string

        X_train, X_test, y_train, y_test = data

        # Feature engineering: Vectorizer. ML models need features, not just whole tweets
        vec = CountVectorizer(analyzer="word", ngram_range=ngram_range)
        print(f"Fitting {sample_type.capitalize()}-sample CV...") if verbose else None
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
        print(f"\nClassification Report [{sample_type.lower()}, {analyzer}, ngram_range{ngram_range}]:\n "
              f"{classification_report(y_test, svm_fitted.predict(X_test_CV), digits=6)}")


""" IMPORT DATA """


def get_data(dev, sample_size, manual_boost):
    random_data = get_random_data()[0:sample_size]
    boosted_topic_data = get_boosted_data(manual_boost)[0:sample_size]
    boosted_wordbank_data = get_boosted_data()[0:sample_size]

    # split data into X, y
    random_splits = split_data(random_data, dev)
    topic_splits = split_data(boosted_topic_data, dev)
    wordbank_splits = split_data(boosted_wordbank_data, dev)

    # return data and identifiers
    return [[random_splits, "random"], [topic_splits, "boosted (topic)"], [wordbank_splits, "boosted (wordbank)"]]


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
            if delimiter is "tab":  # tsv
                splitLine = e.split("\t", 1)
            else:  # default: csv
                splitLine = e.split(",", 1)

            if len(splitLine) is 2:  # else: there's no score, so throw the example out
                data_list.append([float(splitLine[0]), splitLine[1]])

    data = pd.DataFrame(data_list, columns=["score", "comment_text"])
    print(f"Data {data.shape} imported!\n") if verbose else None  # progress indicator

    kaggle_threshold = 0.50  # from Kaggle documentation (see page)

    # create class vector
    data["class"] = 0
    data.loc[data.score >= kaggle_threshold, "class"] = 1

    # remove score vector; replaced by class (bool) vector
    data = data.drop(columns="score")

    return data


# already saved as `.csv`. just import
def get_random_data():
    return read_data("train.random.csv", delimiter="comma")


def get_boosted_data(manual_boost=None):
    data_file = "train.target+comments.tsv"  # only imports dataset once
    data = read_data(data_file, delimiter="tab")

    return boost_data(data, data_file, manual_boost)


""" PROCESS DATA """

# boosts data based on given topics or predefined wordbanks
def boost_data(data, data_file, manual_boost=None):
    print(f"Boosting data...") if verbose else None

    boosted_data = filter_data(data, data_file, manual_boost)
    boosted_data = boosted_data.sample(frac=1)  # shuffle before returning
    return boosted_data


# return data that contains any word in Wordbank
# NOTE: data[data["comment_text"].str.contains("example")] did NOT work so I had to read line-by-line
def filter_data(data, data_file, manual_boost=None):
    """
    data (df):          dataset to filter
    topics ([str]]):    word(s) to filter with. this wordbank bypasses the banks below
    """
    if verbose:
        if manual_boost:
            print(f"Filtering `{data_file} on [{manual_boost}]`...")
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
    print(f"Data filtered to size {filtered_data.shape[0]}.\n") if verbose else None
    return filtered_data


""" SCRIPT CONFIG """
sample_size = 20000  # int
samples = "both"  # "random", "boosted_topic", "boosted_wordbank", or "all"
analyzer = "word"  # "char" or "word"
ngram_range = (1, 1)  # int 2-tuple / couple
gridsearch = True  # bool
dev = False  # bool
manual_boost = None  # ["trump"]  # None, or an array of strings

fit_data(verbose, sample_size, samples, analyzer, ngram_range, gridsearch, manual_boost)
