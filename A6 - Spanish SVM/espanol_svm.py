# LING-X 490 Assignment 6: Spanish SVM
# Dante Razo, drazo, 11/14/2019
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics
import pandas as pd
import random


# Import data; TODO: remove httP://t.co/* links
def get_data():
    data_dir = "./data"
    language = "es"
    fixed = "_fixed"  # empty string if you want to use the original data

    X_train = pd.read_csv(f"{data_dir}/public_development_{language}/train_{language}{fixed}.tsv", sep='\t').iloc[:, 1]
    X_test = pd.read_csv(f"{data_dir}/reference_test_{language}/{language}.tsv", sep='\t').iloc[:, 1]

    y = 2  # HS = "Hate Speech"? making a big assumption here
    y_train = pd.read_csv(f"{data_dir}/public_development_{language}/train_{language}{fixed}.tsv", sep='\t').iloc[:, y]
    y_test = pd.read_csv(f"{data_dir}/reference_test_{language}/{language}.tsv", sep='\t').iloc[:, y]

    return X_train, X_test, y_train, y_test


# Feature engineering: vectorizer
# ML models need features, not just whole tweets
print("COUNTVECTORIZER CONFIG\n----------------------")
analyzer = input("Please enter analyzer: ")
ngram_upper_bound = input("Please enter ngram upper bound(s): ").split()

for i in ngram_upper_bound:
    X_train, X_test, y_train, y_test = get_data()

    vec = CountVectorizer(analyzer=analyzer, ngram_range=(1, int(i)))
    print("\nFitting CV...")
    X_train = vec.fit_transform(X_train)
    X_test = vec.transform(X_test)

    # Shuffle data (keeps indices)
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    # Fitting the model
    print("Training SVM...")
    svm = SVC(kernel="linear", gamma="auto")  # TODO: tweak params
    svm.fit(X_train, y_train)
    print("Training complete.\n")

    """ KERNEL RESULTS gamma="auto", analyzer=word, ngram_range(1,3)
    linear: 0.7254534083802376
    rbf: 0.5872420262664165
    poly: 0.5872420262664165
    sigmoid: 0.5872420262664165
    precomputed: N/A, not supported
    """

    # Testing + results
    rand_acc = sklearn.metrics.balanced_accuracy_score(y_test, [random.randint(0, 1) for x in range(0, len(y_test))])
    print(f"Random/Baseline Accuracy: {rand_acc}")
    print(f"Testing Accuracy ({analyzer}, ngram_range(1,{i})): "
          f"{sklearn.metrics.accuracy_score(y_test, svm.predict(X_test))}".rjust(65))

""" CountVectorizer PARAM TESTING (kernel="linear")
word, ngram_range(1,2):  0.7229518449030644
word, ngram_range(1,3):  0.7254534083802376
word, ngram_range(1,5):  0.7329580988117573
word, ngram_range(1,10): 0.7467166979362101
word, ngram_range(1,20): 0.7367104440275172
char, ngram_range(1,2):  0.6278924327704816
char, ngram_range(1,3):  0.6766729205753595
char, ngram_range(1,5):  0.717948717948718 # Wow!
char, ngram_range(1,10): 0.7304565353345841
char, ngram_range(1,20): 0.734208880550344
"""
