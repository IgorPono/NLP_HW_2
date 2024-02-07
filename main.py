import math
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


def accuracy(predictions, labels):  # compares accuracy with label data
    if len(predictions) != len(labels):
        raise ValueError("Lists must have the same length")

    count = 0
    for pred, label in zip(predictions, labels):
        if pred == label:
            count += 1

    return count / len(labels)


def evaluate_labels(predictions, labels, reviews):  # prints samples that were not accurately guessed
    if len(predictions) != len(labels):
        raise ValueError("Lists must have the same length")

    count = 0
    for idx in range(len(predictions)):
        if predictions[idx] == labels[idx]:
            print("Correct Prediction")
            print("Predicted Label: ", predictions[idx])
            print("Actual Label: ", labels[idx])
            print(idx, ": ", reviews[idx])
            print()
        else:
            print("Incorrect Prediction")
            print("Predicted Label: ", predictions[idx])
            print("Actual Label: ", labels[idx])
            print(idx, ": ", reviews[idx])
            print()


def feature_probability(nb_classifier):
    for word in nb_classifier.vocabulary:
        for label, likelihood in enumerate(nb_classifier.likelihood[word]):
            print(f"Feature: {word}, Class Label: {label}, Likelihood Probability: {likelihood}")


class naive_bayes_classifier:
    def __init__(self):
        self.prior = None
        self.likelihood = None
        self.vocabulary = None  # for testing only

    def fit(self, X, y):
        vec = CountVectorizer(max_features=200)  # edit value to adjust classifier performance

        X_counts = vec.fit_transform(X)  # counts the number of occurrences of each word in the input text

        self.vocabulary = vec.get_feature_names_out()  # pulls out features for evaluation only

        total_samples = X_counts.shape[0]
        self.prior = np.bincount(y) / total_samples
        # calculate probability of either label, should be 50/50 as data
        # was stratified and IMDB has 50/50 pos/neg

        self.likelihood = defaultdict(
            lambda: np.zeros(len(np.unique(y))))  # initialize dictionary that stores likelihood of a word given a class
        for word, idx in vec.vocabulary_.items():
            word_counts = X_counts[:, idx].toarray().flatten()
            for label in np.unique(y):
                samples_with_label = X_counts[y == label].sum()  # sums all words with specific class label
                self.likelihood[word][label] = (word_counts[y == label].sum() + 1) / (samples_with_label + 2)
                # + 1 added to avoid zero probability
                # + 2 added to ensure laplace smoothing works properly

    def predict(self, X):
        predictions = []
        for review in X:
            prob_neg = math.log(self.prior[0])
            prob_pos = math.log(self.prior[1])
            for word in review.split():
                if word in self.likelihood:
                    prob_neg += math.log(self.likelihood[word][0])
                    prob_pos += math.log(self.likelihood[word][1])
            if prob_pos > prob_neg:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions


df = pd.read_csv("./IMDB Dataset.csv")  # load IMDB dataset

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # load the stop words dictionary

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()


def remove_stopwords(review):
    tokens = review.split()  # split individual review based on white space
    filtered_text = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)  # join the tokens based with white space


df['review'] = df['review'].apply(remove_stopwords)  # apply stop_words function to all reviews

encoder = LabelEncoder()
df['sentiment'] = encoder.fit_transform(df['sentiment'])  # encode labels in place

train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['sentiment'])
val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df['sentiment'])  # split into train/val/test

train_reviews = train_df['review'].values
train_sentiments = train_df['sentiment'].values

val_reviews = val_df['review'].values
val_sentiments = val_df['sentiment'].values

test_reviews = test_df['review'].values
test_sentiments = test_df['sentiment'].values

naive_bayes = naive_bayes_classifier()
naive_bayes.fit(train_reviews, train_sentiments)
val_prediction = naive_bayes.predict(val_reviews)

test_prediction = naive_bayes.predict(test_reviews)

print((evaluate_labels(val_prediction, val_sentiments, val_reviews)))
print(evaluate_labels(test_prediction, test_sentiments, test_reviews))
print()

print("Validation set prediction accuracy: ", accuracy(val_prediction, val_sentiments))
print("Test set prediction accuracy: ", accuracy(test_prediction, test_sentiments))

feature_probability(naive_bayes)
