# LING-X 490 FA19 Final: Boosted Kaggle SVM
# Dante Razo, drazo, 12/18/2019
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics
import pandas as pd
import random


# Import data; TO CONSIDER: remove http://t.co/* links, NEWLINE_TOKEN
# original Kaggle dataset: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
def get_data(verbose, sample_types, sample_size=10000):
    data_dir = "./data"
    data = pd.read_csv(f"{data_dir}/toxicity_annotated_comments.tsv", sep='\t', header=0)
    data.sample(frac=1)  # shuffle data

    to_return = []

    # sampled datasets
    if sample_size > 66838:
        sample_size = 66838  # bound; number of entries in dataset with abusive language matches

    print(f"Boosting data...") if verbose else None
    boosted_data = boost_data(data).sample(frac=1)[0:sample_size]  # < 66838
    print(f"Boosting complete.") if verbose else None
    random_sample = data.sample(sample_size)  # ensures that both sets are the same size

    print(f"len: {len(boosted_data)}")

    for s in sample_types:
        if s is "Boosted":
            data = boosted_data
        elif s is "Random":
            data = random_sample

        train = data.loc[data['split'] == "train"]
        test = data.loc[data['split'] == "test"]
        dev = data.loc[data['split'] == "dev"]

        X_train = train.iloc[:, 1]
        X_test = test.iloc[:, 1]
        X_dev = dev.iloc[:, 1]  # ignoring dev for now...

        y = 3  # assumes that 'logged_in' is the class feature
        y_train = train.iloc[:, y] * 1
        y_test = test.iloc[:, y] * 1
        y_dev = dev.iloc[:, y] * 1

        to_return.append([X_train, X_test, X_dev, y_train, y_test, y_dev])

    return to_return


# boosting; filters on abusive language
def boost_data(data):
    lexicon_dir = "./lexicon"
    version = "base"  # or "expanded"
    df = pd.read_csv(f"{lexicon_dir}/{version}Lexicon.txt", sep='\t', header=None)
    lexicon = pd.DataFrame(columns=["word", "part", "hate"])

    # split into three features
    lexicon[["word", "part"]] = df[0].str.split('_', expand=True)
    lexicon["hate"] = df[1]

    # list of abusive words
    hate = list(lexicon[lexicon["hate"]]["word"])
    concat_hate = '|'.join(hate)

    # data containing abusive words
    abusive_data = data[data["comment"].str.contains(concat_hate)]

    return abusive_data


# Feature engineering: vectorizer
# ML models need features, not just whole tweets
development = True  # flag; ask for input if False
if development:
    analyzer, ngram_upper_bound, sample_size = ["word", [3], 1000]  # default values for quick fits
else:
    print("COUNTVECTORIZER CONFIG\n----------------------")
    analyzer = input("Please enter analyzer: ")
    ngram_upper_bound = input("Please enter ngram upper bound(s): ").split()
    sample_size = input("Please enter sample size (< 66839): ")

sample_types = ["Boosted", "Random"]

for i in ngram_upper_bound:
    for t in range(0, len(sample_types)):
        verbose = True  # print statement flag
        X_train, X_test, X_dev, y_train, y_test, y_dev = get_data(verbose, sample_types, sample_size)[t]

        vec = CountVectorizer(analyzer=analyzer, ngram_range=(1, int(i)))
        print(f"\nFitting {sample_types[t]}-sample CV...") if verbose else None
        X_train = vec.fit_transform(X_train)
        X_test = vec.transform(X_test)

        # Shuffle data (keeps indices)
        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)

        # Fitting the model
        print(f"Training {sample_types[t]}-sample SVM...") if verbose else None
        svm = SVC(kernel="linear", gamma="auto")  # TODO: tweak params
        svm.fit(X_train, y_train)
        print(f"Training complete.") if verbose else None

        # Testing + results
        acc_score = sklearn.metrics.accuracy_score(y_test, svm.predict(X_test))

        print(f"\nAccuracy for {sample_types[t]}-sample ({analyzer}, ngram_range(1,{i}): {acc_score}")

""" RESULTS & DOCUMENTATION
# BOOSTED KERNEL TESTING (gamma="auto", analyzer=word, ngram_range(1,3))
## NOTES: 
linear:  
rbf:     
poly:    
sigmoid: 
precomputed: N/A, not supported

# RANDOM KERNEL TESTING (gamma="auto", analyzer=word, ngram_range(1,3))
## NOTES: 
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
