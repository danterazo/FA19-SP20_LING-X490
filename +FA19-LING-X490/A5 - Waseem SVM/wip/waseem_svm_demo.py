"""
#Let's do some NLP now!

For this script, we'll be adding in:
- dataset reading
- feature extraction
"""

import sklearn
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer


def extract_features(data):
    """
    data is a list of strings here where each string represents a document
    that we're trying to determine the sentiment of.

    This is a very simple feature extraction method that just takes a
    list of positive words, a list of negative words and then gets the count
    of both of these word groups in the given string. An additional feature is
    used to signal if negation was present.
    """
    positive_words = ['good', 'great', 'fantastic', 'excellent', 'worthwhile']
    negative_words = ['bad', 'horrible', 'boring', 'terrible', 'uninteresting']
    res = []
    for entry in data:
        toked_entry = word_tokenize(entry)
        neg_count = 0
        pos_count = 0
        negation_flag = 0
        word_counts = Counter(toked_entry)
        for word in negative_words:
            if word in word_counts.keys():
                neg_count += word_counts[word]

        for word in positive_words:
            if word in word_counts.keys():
                pos_count += word_counts[word]

        if 'not' in toked_entry:
            negation_flag = 1

        res_row = [neg_count, pos_count, negation_flag]
        res.append(res_row)

    # This changes the list of lists into a more compact array
    # representation that only stores non-zero values
    res = csr_matrix(res)
    return res


def read_dataset(file_path):
    """
    File_path should be a string that represents the filepath
    where the movie dataset can be found

    This returns an array of strings and an array of labels
    """
    neg_data = []
    pos_data = []
    for root, dirs, files in os.walk(file_path + "/neg"):
        for file_name in files:
            fp = open(os.path.join(root, file_name))
            neg_data.append(fp.read())

    for root, dirs, files in os.walk(file_path + "/pos"):
        for file_name in files:
            fp = open(os.path.join(root, file_name))
            pos_data.append(fp.read())

    neg_labels = np.repeat(-1, len(neg_data))
    pos_labels = np.repeat(1, len(pos_data))
    labels = np.concatenate([neg_labels, pos_labels])
    data = neg_data + pos_data
    return data, labels


def main():
    # Read in dataset
    print("Reading in dataset...")
    train_text_data, train_Y = read_dataset("aclImdb/train")
    test_text_data, test_Y = read_dataset("aclImdb/test")
    print(Counter(test_Y))
    # Now we need to extract features from the text data
    print("Extracting features...")
    # To run a bag-of-words feature extractor in sklearn we first initialize the vectorizer
    # Check out the documentation to see all the different settings that can be used
    # In this case, we are doing analysis at the word level with unigrams, bigrams and trigrams
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3))
    # Now we run fit transform on the vectorizer
    # this will figure out what the unique n-grams are and return the feature matrix given these
    train_X = vectorizer.fit_transform(train_text_data)
    # when we run over the test data though, we don't want to find new unique n-grams
    # This time, instead of running fit_transform, we simply run transform
    test_X = vectorizer.transform(test_text_data)

    # if the following two lines are uncommented, my feature extraction function
    # will be used to get our training and test data instead
    # train_X = extract_features(train_text_data)
    # test_X = extract_features(test_text_data)
    params = {'n_estimators': [10, 20, 30, 100],
              'criterion': ['gini', 'entropy']}
    rf_model = rf(n_jobs=1)
    model = GridSearchCV(rf_model, params, n_jobs=4, cv=5)
    print("Training...")
    model.fit(train_X, train_Y)
    preds = model.predict(test_X)
    print(classification_report(test_Y, preds, digits=6))


if __name__ == '__main__':
    main()

# Modify extract_features to scale the positive and negative counts 
# by the length of the document  
# Make your own feature extraction method using extract_features as a template
