import pandas as pd
import numpy as np
import nltk
import re
import os
import tensorflow as tf
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from os.path import exists
from pathlib import Path


class Classification:

    def __init__(self):

        self.data = []
        self.target = []
        self.X = []
        self.y = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

    def load_data(self):

        dataset_path = './data/spam.csv'
        file_exists = os.path.exists(dataset_path)

        if (file_exists):
            self.data = pd.read_csv(dataset_path, encoding = 'latin-1').iloc[:, :2].values
        else:
            print('path not found')


    def lang_processing(self):
        
        le = LabelEncoder()
        self.target = le.fit_transform(self.data[:,0])

        unique, count = np.unique(self.target, return_counts=True)

        # plt.xticks(unique, unique)
        # plt.bar(unique, count, color=['blue','orange'])
        # plt.show()

        Stopwords = stopwords.words('english')
        stemmer = PorterStemmer()
        email = self.data[:,1]
        email = [re.sub("[^a-zA-Z]"," ", e) for e in email]
        strings = np.char.split(np.char.lower(email))
        words = [[stemmer.stem(word) for word in string if word not in set(Stopwords)] for string in strings ]
        sentence = [' '.join(row) for row in words]
        self.X = sentence


    def split_data(self):

        cv = CountVectorizer()
        tfidf = TfidfTransformer()
        self.y = tf.keras.utils.to_categorical(self.target,2)
        self.X = cv.fit_transform(self.X).toarray()

        self.X = tfidf.fit_transform(self.X).toarray()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=12)
        print('train vs test: {}, {}'.format(self.X_train.shape[0], self.X_test.shape[0]))



if __name__ == "__main__":
    c = Classification()
    c.load_data()
    c.lang_processing()
    c.split_data()
