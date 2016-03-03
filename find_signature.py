#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]

### Classifier here
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(features_train,labels_train)

accuracy = dt.score(features_test,labels_test)
print("\n Accuracy of Model: %0.3F %%" % (accuracy*100))

# Find the top feature in the decision tree and its relative importance
top_feature = dt.feature_importances_[dt.feature_importances_ > 0.2]

import numpy as np
idx = np.where(dt.feature_importances_ > 0.2)

print("\n Value of most important feature: %0.4F " % top_feature)
print("\n Number of most important feature: %0.0F " % idx[0][0] )

# What is the word that is causing the trouble
vocab_list = vectorizer.get_feature_names()
print("\n Word causing most discrimination on the decision tree: %s" % vocab_list[idx[0][0]])