# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 12:45:55 2018

@author: SumedhPC
"""

#Text Classification

#Importing the libraries
import numpy as np
import re
import pickle
import nltk 
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')

#Importing the dataset
reviews = load_files('txt_sentoken/')
X, y = reviews.data, reviews.target

#Storing as pickle files
with open('X.pickle', 'wb') as f:
    pickle.dump(X, f)
    
with open('y.pickle', 'wb') as f:
    pickle.dump(y,f)

#Unpickling a dataset
with open('X.pickle', 'rb') as f:
    pickle.load(f)

with open('y.pickle', 'rb') as f:
    pickle.load(f)    

#Creating the corpus
corpus = []
for i in range(0, len(X)):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+', ' ', review)
    review = re.sub(r'^[a-z]\s+', ' ', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)

#BOW model
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()    

#BOW model to TF-IDF model
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

#Tfidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()    


#Splitting the data into Test and Train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Model1
#Using Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)     

sent_pred = classifier.predict(X_test)    
    
#Generating the confusion matrix to prdict accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, sent_pred)

#Accuracy
acc = (cm[0][0] + cm[1][1]) / 4
print(acc)   
#The accuracy with logistic regression is 84.75%

#Model2
#Using SVM

#Splitting the data into Test and Train
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.22, random_state = 0)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier1 = SVC(kernel = 'linear')
classifier1.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred = classifier.predict(X_test1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test1, y_pred)
cm1

#Accuracy
acc1 = (cm1[0][0] + cm1[1][1]) / 4
print(acc1)   
#The accuracy with SVM is 94.75%



#Pickling the classifier
with open('classifier.pickle', 'wb') as f:
    pickle.dump(classifier, f)

#Pickling the vectorizer    
with open('tfidfmodel.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)
    
#Unpickling the classifier and vectorizer
with open('classifier.pickle', 'rb') as f:
    clf = pickle.load(f)
    
with open('tfidfmodel.pickle', 'rb') as f:
    tfidf = pickle.load(f)

#Testing the model on some samples        
sample = ["You are a good man, have a good life"]
sample = tfidf.transform(sample).toarray()
print(clf.predict(sample))

sample1 = ["You are not a good man"]
sample1 = tfidf.transform(sample1).toarray()
print(clf.predict(sample1))





                                    
    
    
