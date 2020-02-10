# LING-X 490 FA19 Final: Boosted Kaggle SVM
# Dante Razo, drazo; due 12/18/2019 @ 11:59pm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import sklearn.metrics
import pandas as pd

pd.options.mode.chained_assignment = None  # suppress SettingWithCopyWarning


def get_data(verbose, boost_threshold, sample_types, sample_size=10000):  # TODO: increase sample size
    data_dir = "../../data/kaggle_data"  # common directory for all datasets
    dataset = "train"  # 'test' for classification problem
    data = pd.read_csv(f"{data_dir}/{dataset}.csv", sep=',', header=0)  # import Kaggle data
    data = data.iloc[:, 1:3]  # eyes on the prize (only focus on important columns)
    kaggle_threshold = 0.5  # from Kaggle documentation (see page)
    dev = True  # set to FALSE when its time to validate `train` dataset
    to_return = []  # this function returns a list of lists. Each inner list contains `X` and `y`

    # create class vector
    data["class"] = 0
    data.loc[data["target"] >= kaggle_threshold, ["class"]] = 1

    # remove old class vector ('target')
    data = data.loc[:, data.columns != 'target']

    # sampled datasets
    data_len = len(data)
    if sample_size > data_len or sample_size < 1:
        sample_size = data_len  # bound

    # boosted_data = boost_data(data[0:sample_size], boost_threshold, verbose) # TODO: reimplement
    random_sample = data.sample(frac=1).sample(len(data))[0:sample_size]  # shuffle first, then pick out `n` entries

    for s in sample_types:
        if s is "boosted":
            # data = boosted_data.sample(frac=1)  # reshuffle
            pass  # debugging, to remove
        elif s is "random":
            data = random_sample.sample(frac=1)  # reshuffle

        X = data.loc[:, data.columns != "class"]
        y = data.loc[:, data.columns == "class"]

        # train: 60%, dev: 20%, test: 20%
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y,
                                                                                    test_size=0.2,
                                                                                    shuffle=True,
                                                                                    random_state=42)

        X_train, X_dev, y_train, y_dev = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                                  test_size=0.25,
                                                                                  shuffle=True,
                                                                                  random_state=42)

        to_return.append([X_train, X_dev, y_train, y_dev]) if dev \
            else to_return.append([X_train, X_test, y_train, y_test])  # use dev sets if dev=TRUE

    # [[boosted X, y], [randomly sampled X, y]]
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
    data["count"] = 0  # loc to suppress SettingWithCopyWarning

    print(f"data boosted headers: {data.columns}")  # debugging

    # data containing abusive words
    for i in range(0, len(data)):
        words = data.loc["comment_text"].iloc[i].split(" ")  # split comment into words

        for word in words:
            if word in hate:
                data.loc["count"][i] += 1  # increment

    abusive_data = data.loc[data["count"] >= boost_threshold]
    print(f"Boosting complete.") if verbose else None

    print(f"sum: {sum(data['count'])}; shape: {abusive_data.shape}")
    print(f"data shape: {data.shape}")

    return abusive_data.iloc[:, 0:7]


""" CONFIGURATION """
mode = "nohup"  # mode switch: "quick" / "nohup" / "user"
verbose = True  # print statement flag
sample_type = ["boosted", "random"]  # do both types of sampling

if mode is "quick":  # for development. quick fits
    print("DEVELOPMENT MODE ----------------------")
    analyzer, ngram_upper_bound, sample_size, boost_threshold = ["word", [3], 1000, 1]
    sample_type = ["random"]

elif mode is "nohup":  # nohup mode. hard-code inputs here, switch the mode above, then run!
    print("NOHUP MODE -------------------------")
    analyzer = "word"
    ngram_upper_bound = [3]
    sample_size = 1804874  # try: 50000. max: 1804874
    boost_threshold = 1
    verbose = False

else:  # user-interactive mode. Good for running locally... not good for nohup
    print("COUNTVECTORIZER CONFIG\n----------------------")
    analyzer = input("Please enter analyzer: ")
    ngram_upper_bound = input("Please enter ngram upper bound(s): ").split()
    sample_size = input("Please enter sample size (< 66839): ")
    boost_threshold = input("Please enter the hate speech threshold: ")  # num of abusive words each entry must contain

""" MAIN """
data = get_data(verbose, boost_threshold, sample_type, sample_size)  # array of data. [[boosted X,y], [random X,y]]

# allows different ngram bounds w/o `Pipeline`
for i in ngram_upper_bound:

    # Try the current parameters with each sampling type
    for t in range(0, len(sample_type)):
        X_train, X_test, y_train, y_test = data[t]

        # Feature engineering: Vectorizer. ML models need features, not just whole tweets
        vec = CountVectorizer(analyzer="word", ngram_range=(1, 1))
        print(f"\nFitting {sample_type[t].capitalize()}-sample CV...") if verbose else None
        X_train_CV = vec.fit_transform(X_train["comment_text"])
        X_test_CV = vec.transform(X_test["comment_text"])

        # Fitting the model
        print(f"Training {sample_type[t].capitalize()}-sample SVM...") if verbose else None
        svm_model = SVC(kernel="linear", gamma="auto")
        svm_params = {'C': [0.1, 1, 10, 100, 1000],  # regularization param
                      'gamma': ["scale", "auto", 1, 0.1, 0.01, 0.001, 0.0001],  # kernel coefficient (R, P, S)
                      'kernel': ["linear", "poly", "rbf", "sigmoid"]}  # SVM kernel (precomputed not supported)
        svm_gs = GridSearchCV(svm_model, svm_params, n_jobs=4, cv=5)
        svm_gs.fit(X_train_CV, y_train.values.ravel())
        print(f"Training complete.") if verbose else None

        # Testing + results
        nl = "" if mode is "nohup" else "\n"  # groups results together when training
        print(f"{nl}Classification Report [{sample_type[t].lower()}, {analyzer}, ngram_range(1,{i})]:\n "
              f"{classification_report(y_test, svm_gs.predict(X_test_CV), digits=6)}")

""" RESULTS & DOCUMENTATION
# KERNEL TESTING (RANDOM, size=50000, gamma="auto", analyzer=word, ngram_range(1,3))
linear:  
rbf:     
poly:    
sigmoid: 
precomputed: N/A, not supported

# BOOSTED CountVectorizer PARAM TESTING (kernel="linear")
word, ngram_range(1,3): 
char, ngram_range(1,3):  

# RANDOM CountVectorizer PARAM TESTING (kernel="linear")
word, ngram_range(1,3):  
char, ngram_range(1,3):  

## Train start (all): 
## Train end (word):  
## Train kill (all):  
"""
