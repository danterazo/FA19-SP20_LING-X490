# LING-X 490 Assignment 5: Waseem SVM
# Dante Razo, drazo, 10/30/2019
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics
import random


# Import data; TODO: remove httP://t.co/* links
def get_data():
    data_dir = "./data/"
    a = open(f"{data_dir}X_train", "r", encoding="utf-8")
    X_train = a.read().splitlines()

    a = open(f"{data_dir}X_test", "r", encoding="utf-8")
    X_test = a.read().splitlines()

    a = open(f"{data_dir}y_train", "r", encoding="utf-8")
    y_train = a.read().splitlines()
    for i in range(0, len(y_train)):
        y_train[i] = int(y_train[i])

    a = open(f"{data_dir}y_test", "r", encoding="utf-8")
    y_test = a.read().splitlines()
    for i in range(0, len(y_test)):
        y_test[i] = int(y_test[i])

    a.close()
    return X_train, X_test, y_train, y_test


# Feature engineering: vectorizer
# ML models need features, not just whole tweets
print("COUNTVECTORIZER CONFIG\n----------------------")
analyzer = input("Please enter analyzer: ")
ngram_upper_bound = input("Please enter ngram upper bound(s): ").split()

for i in ngram_upper_bound:
    X_train, X_test, y_train, y_test = get_data()
    verbose = False  # print statement flag

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
# KERNEL RESULTS (gamma="auto", analyzer=word, ngram_range(1,3))
linear:  0.8549618320610687
rbf:     0.6844783715012722
poly:    0.6844783715012722
sigmoid: 0.6844783715012722
precomputed: N/A, not supported

# CV PARAM TESTING (kernel="linear")
word, ngram_range(1,2):  0.8606870229007634
word, ngram_range(1,3):  0.8549618320610687
word, ngram_range(1,5):  0.8473282442748091
word, ngram_range(1,10): 0.8358778625954199
word, ngram_range(1,20): 0.8326972010178118
char, ngram_range(1,2):  0.8225190839694656
char, ngram_range(1,3):  0.8206106870229007
char, ngram_range(1,5):  0.8473282442748091
char, ngram_range(1,10): 0.8473282442748091
char, ngram_range(1,20): 0.8575063613231552
"""
