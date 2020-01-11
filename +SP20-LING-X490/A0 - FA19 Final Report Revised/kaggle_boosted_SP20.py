# LING-X 490 FA19 Final: Boosted Kaggle SVM
# Dante Razo, drazo; due 12/18/2019 @ 11:59pm
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics
from sklearn.metrics import accuracy_score
import pandas as pd

pd.options.mode.chained_assignment = None  # suppress SettingWithCopyWarning

# Import data; TO CONSIDER: remove http://t.co/* links, :NEWLINE_TOKEN:
# original Kaggle dataset: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
def get_data(verbose, boost_threshold, sample_types, sample_size=10000):
    data_dir = "../../Data/kaggle_data"
    dataset = "train"  # test is classification
    data = pd.read_csv(f"{data_dir}/{dataset}.csv", sep=',')
    # data = data.iloc[:, 1:7]  # remove categorical data
    kaggle_threshold = 0.5  # from documentation

    print(f"data shape: {data.shape}")  # debugging

    # class
    data["class"] = 0
    data.loc[data["target"] >= kaggle_threshold, ["class"]] = 1

    data.sample(frac=1)  # shuffle data
    to_return = []

    # sampled datasets
    data_len = len(data)
    if sample_size > data_len or sample_size < 1:
        sample_size = data_len  # bound

    boosted_data = boost_data(data[0:sample_size], boost_threshold, verbose)
    random_sample = data.sample(len(data))[0:sample_size]  # not the same size

    for s in sample_types:
        if s is "boosted":
            data = boosted_data.sample(frac=1)  # reshuffle
        elif s is "random":
            data = random_sample.sample(frac=1)  # reshuffle

        X = data.loc[:, data.columns != "class"]
        y = data.loc[:, data.columns == "class"]

        print(f"X shape: {X.shape}; Y shape: {y.shape}")  # debug
        # print(f"X:\n {X[0:3]}\n\ny: \n{y[0:3]}")

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.75,
                                                                                    shuffle=True,
                                                                                    random_state=42)

        # y_train = np.ravel(y_train.to_numpy())
        # y_test = np.ravel(y_test.to_numpy())
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        to_return.append([X_train, X_test, y_train, y_test])

    return to_return


# boosting; filters on abusive language
def boost_data(data, boost_threshold, verbose):
    print(f"Boosting data...") if verbose else None
    lexicon_dir = "./lexicon"
    version = "base"  # or "expanded"
    df = pd.read_csv(f"{lexicon_dir}/{version}Lexicon.txt", sep='\t', header=None)
    lexicon = pd.DataFrame(columns=["word", "part", "hate"])

    # split into three features
    lexicon[["word", "part"]] = df[0].str.split('_', expand=True)
    lexicon["hate"] = df[1]

    # list of abusive words
    hate = list(lexicon[lexicon["hate"]]["word"])

    # add abusive word count feature to data
    data["count"] = 0

    # data containing abusive words
    for i in range(0, len(data)):
        words = data["comment_text"][i].split(" ")  # split comment into words

        for word in words:
            if word in hate:
                data["count"][i] += 1  # increment

    abusive_data = data.loc[data["count"] >= boost_threshold]
    print(f"Boosting complete.") if verbose else None

    print(f"sum: {sum(data['count'])}; shape: {abusive_data.shape}")
    print(f"data shape: {data.shape}")

    return abusive_data.iloc[:, 0:7]


# Feature engineering: vectorizer
# ML models need features, not just whole tweets
mode = "dev"  # mode switch: "dev" | "train" | "user"
verbose = True  # print statement flag
if mode is "dev":
    print("DEVELOPMENT MODE ----------------------")
    analyzer, ngram_upper_bound, sample_size, boost_threshold = ["word", [3], 1000, 1]  # default values for quick fits
elif mode is "train":
    print("TRAINING MODE -------------------------")
    analyzer = "word"  # default values for consistent quality fits
    ngram_upper_bound = [2, 3, 5, 10]
    sample_size = 50000  # max: 1804874
    boost_threshold = 1
    verbose = False
else:
    print("COUNTVECTORIZER CONFIG\n----------------------")
    analyzer = input("Please enter analyzer: ")
    ngram_upper_bound = input("Please enter ngram upper bound(s): ").split()
    sample_size = input("Please enter sample size (< 66839): ")
    boost_threshold = input("Please enter the hate speech treshold: ")  # num of abusive words each entry must contain

sample_types = ["boosted", "random"]
data = get_data(verbose, boost_threshold, sample_types, sample_size)

for i in ngram_upper_bound:
    for t in range(0, len(sample_types)):
        X_train, X_test, y_train, y_test = data[t]

        # X_train = X_train
        # X_test = X_test
        # y_train = list(y_train["class"])
        # y_test = list(y_test["class"])

        # print(X_train.head())

        print(
            f"shapes:\n Xtr: {X_train.shape}, Xte: {X_test.shape}, ytr: {len(y_train)}, yte: {len(y_test)}")  # debug

        vec = CountVectorizer(analyzer=analyzer, ngram_range=(1, int(i)))
        print(f"\nFitting {sample_types[t].capitalize()}-sample CV...") if verbose else None
        X_train = vec.fit_transform(X_train)
        X_test = vec.transform(X_test)

        # Shuffle data (keeps indices)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        X_test, y_test = sklearn.utils.shuffle(X_test, y_test)

        # Fitting the model
        print(f"Training {sample_types[t].capitalize()}-sample SVM...") if verbose else None
        svm = SVC(kernel="linear", gamma="auto")  # TODO: tweak params
        svm.fit(X_train, y_train)
        print(f"Training complete.") if verbose else None

        # Testing + results
        acc_score = accuracy_score(y_test, svm.predict(X_test))
        nl = "" if mode is "train" else "\n"  # groups results together when training
        print(f"{nl}Accuracy [{sample_types[t].lower()}, {analyzer}, ngram_range(1,{i})]: {acc_score}")

""" RESULTS & DOCUMENTATION
# KERNEL TESTING (RANDOM, size=50000, gamma="auto", analyzer=word, ngram_range(1,3))
linear:  
rbf:     
poly:    
sigmoid: 
precomputed: N/A, not supported

# BOOSTED CountVectorizer PARAM TESTING (kernel="linear")
word, ngram_range(1,2):  
word, ngram_range(1,3):  
word, ngram_range(1,5):  
word, ngram_range(1,10): 
char, ngram_range(1,2):  
char, ngram_range(1,3):  
char, ngram_range(1,5):  
char, ngram_range(1,10): 

# RANDOM CountVectorizer PARAM TESTING (kernel="linear")
word, ngram_range(1,2):  
word, ngram_range(1,3):  
word, ngram_range(1,5):  
word, ngram_range(1,10): 
char, ngram_range(1,2):  
char, ngram_range(1,3):  
char, ngram_range(1,5):  
char, ngram_range(1,10): 

## Train start (all): 
## Train end (word):  
## Train kill (all):  
"""
