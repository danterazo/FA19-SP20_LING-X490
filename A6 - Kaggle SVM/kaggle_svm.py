# LING-X 490 Assignment 6: Kaggle SVM
# Dante Razo, drazo, 11/14/2019
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics
import pandas as pd
import random


# Import data; TODO: remove httP://t.co/* links
# original Kaggle dataset: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
def get_data():
    # TODO: split feature = "test" or "train"
    data_dir = "./data"


    """
    X_train = pd.read_csv(f"{data_dir}/public_development_{language}/train_{language}{fixed}.tsv", sep='\t').iloc[:, 1]
    X_test = pd.read_csv(f"{data_dir}/reference_test_{language}/{language}.tsv", sep='\t').iloc[:, 1]

    y = 2  # HS = "Hate Speech"? making a big assumption here
    y_train = pd.read_csv(f"{data_dir}/public_development_{language}/train_{language}{fixed}.tsv", sep='\t').iloc[:, y]
    y_test = pd.read_csv(f"{data_dir}/reference_test_{language}/{language}.tsv", sep='\t').iloc[:, y]
    """

    return X_train, X_test, y_train, y_test


# Feature engineering: vectorizer
# ML models need features, not just whole tweets
print("COUNTVECTORIZER CONFIG\n----------------------")
analyzer = input("Please enter analyzer: ")
ngram_upper_bound = input("Please enter ngram upper bound(s): ").split()

for i in ngram_upper_bound:
    X_train, X_test, y_train, y_test = get_data()
    verbose = True  # print statement flag

    vec = CountVectorizer(analyzer=analyzer, ngram_range=(1, int(i)))
    print("\nFitting CV...") if verbose else None
    X_train = vec.fit_transform(X_train)
    X_test = vec.transform(X_test)

    # Shuffle data (keeps indices)
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    # Fitting the model
    print("Training SVM...") if verbose else None
    svm = SVC(kernel="linear", gamma="auto")  # TODO: tweak params
    svm.fit(X_train, y_train)
    print("Training complete.") if verbose else None

    # Testing + results
    rand_acc = sklearn.metrics.balanced_accuracy_score(y_test, [random.randint(0, 1) for x in range(0, len(y_test))])
    acc_score = sklearn.metrics.accuracy_score(y_test, svm.predict(X_test))

    print(f"\nResults for ({analyzer}, ngram_range(1,{i}):")
    print(f"Baseline Accuracy: {rand_acc:}")  # random
    print(f"Testing Accuracy:  {acc_score:}")

""" RESULTS & DOCUMENTATION
# KERNEL TESTING gamma="auto", analyzer=word, ngram_range(1,3)
linear:  
rbf:     
poly:    
sigmoid: 
precomputed: N/A, not supported

# CountVectorizer PARAM TESTING (kernel="linear")
word, ngram_range(1,2):  
word, ngram_range(1,3):  
word, ngram_range(1,5):  
word, ngram_range(1,10): 
word, ngram_range(1,20): 
char, ngram_range(1,2):  
char, ngram_range(1,3):  
char, ngram_range(1,5):  
char, ngram_range(1,10): 
char, ngram_range(1,20): 
"""
