import os
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
from textblob import TextBlob
import pickle
import numpy as np
from sklearn.metrics import log_loss
import time
from tqdm import tqdm
from flask import Flask, jsonify, request
from flask import render_template
import joblib
from bs4 import BeautifulSoup
import flask

app = Flask(__name__)

def clean_text(text):
    desc = re.sub(r"\'m,"," am", str(text))
    desc = re.sub(r"\won't", "will not", str(text))
    desc = re.sub(r"can\'t", "can not", str(text))
    desc = re.sub(r"\'ll"," will", str(text))
    desc = re.sub(r"\'re", "are", str(text))
    desc = re.sub(r"\'ve", "have", str(text))
    desc = re.sub('[^0-9a-zA-Z +#]', '', str(text))
    desc = desc.lower()
    desc = desc.split()
    desc = [w for w in desc if w not in set(stopwords.words('english'))]
    joined = " ".join(desc)
    return joined

def sentiment(df):
    sentiment_score = [round(TextBlob(article).sentiment.polarity,3) for article in df['Description'].values]
    df['sentiment_score'] = sentiment_score   


pos_dic = {'noun' : ['NN', 'NNS','NNP', 'NNPS'],
           'verb' : ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
           'adj' : ['JJ', 'JJR', 'JJS'],
           'adv' : ['RB', 'RBR', 'RBS', 'WRB']}

def pos_check(x,flag):
    cnt = 0 
    try :
        w = TextBlob(x)
        for tup in w.tags :
            pos = list(tup)[1]
            if pos in pos_dic[flag]:
                cnt += 1
    except :
        pass
    return cnt

def tfidfw2v(text):
    tfidf_w2v_vector = []
    for sentence in text :
        vector = np.zeros(50)
        tfidf_weight = 0
        for word in sentence.split():
            if (word in glove_words) and (word in tfidf_words):
                vec = model_vec[word]
                tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split()))
                vector += (vec * tf_idf)
                tfidf_weight += tf_idf
        if tfidf_weight != 0 :
                vector /= tfidf_weight
        tfidf_w2v_vector.append(vector)
    return tfidf_w2v_vector

class CustomStacking(object):
    def __init__(self, k, list_of_clf, list_of_meta):
        self.k = k
        self.list_of_clf = list_of_clf
        self.list_of_meta = list_of_meta
    def train_base(self, train_data):
        X = train_data[0]
        Y = train_data[1]
        base_clf = []
        len_clf = len(self.list_of_clf)
        for i in tqdm(range(self.k)):
            idx = np.random.randint(0, len_clf)
            classifier = self.list_of_clf[idx]
            classifier.fit(X,Y)
            base_clf.append(classifier)
        print ("Base Model has been trained")
        self.base_clf = base_clf
    def train_meta(self, X,Y):
        yl = list(y)
        df = pd.DataFrame()
        df['label'] = yl
        for i,j in enumerate(self.base_clf):
            pred = j.predict(X)
            df[i] = pred
        target = df['label'].values
        target = target.astype('int')
        predicted = df.drop(['label'], axis = 1)
        predicted = np.array(predicted)
        ## training meta clf
        len_clf = len(self.list_of_meta)
        idx = np.random.randint(0, len_clf)
        classifier = self.list_of_meta[idx]
        self.meta_clf = classifier
        self.meta_clf.fit(predicted, target)
        print ("Meta Classifier has been trained")
    def evaluation(self, X,Y):
        yl = list(Y)
        dataframe = pd.DataFrame()
        dataframe['label'] = yl
        for i,j in enumerate(self.base_clf):
            pred = j.predict(X)
            dataframe[i] = pred
        target = dataframe['label'].values
        target = target.astype('int')
        predicted = dataframe.drop(['label'], axis = 1)
        predicted = np.array(predicted)
        ## accuracy
        pred = self.meta_clf.predict(predicted)
        ## log loss
        prob = self.meta_clf.predict_proba(predicted)
        logloss = log_loss(target,prob)
        print ("log loss is {}".format(logloss))
    def predict(self, datapoint):
        dataframe = pd.DataFrame()
        for i,j in enumerate(self.base_clf):
            pred = j.predict(datapoint)
            dataframe[i] = pred
        predicted = dataframe
        predicted = np.array(predicted)
        pred = self.meta_clf.predict(predicted)
        return pred

class Sklearnhelper(object):
    def __init__(self, clf, params = None):
        self.clf = clf(**params)
    def fit(self, x, y):
        return self.clf.fit(x,y)
    def predict(self,x):
        return self.clf.predict(x)
    def predict_proba(self,x):
        return self.clf.predict_proba(x)

def function_one(datapoint):
    df = pd.DataFrame()
    df['Description'] = datapoint
    sentiment(df)
    df['noun_count'] = df['Description'].apply(lambda x: pos_check(x, 'noun'))
    df['verb_count'] = df['Description'].apply(lambda x: pos_check(x, 'verb'))
    df['adj_count'] = df['Description'].apply(lambda x: pos_check(x, 'adj'))
    df['adv_count'] = df['Description'].apply(lambda x: pos_check(x, 'adv'))
    tfidf_valv = tfidfw2v(df['Description'].values)
    df.drop(['Description'], inplace = True, axis = 1)
    tfidf_valv = np.array(tfidf_valv)
    x_tr = np.hstack((tfidf_valv, df))
    for i in x_tr :
        i = i.reshape(1,-1)
        pred = cst.predict(i)
        return pred

dictionary = pickle.load(open("dictionary_final.pickle", "rb", -1))
tfidf_words = pickle.load(open("tfwords_final.pickle", "rb", -1))
glove_words = pickle.load(open("glove_final.pickle", "rb", -1))
model_vec = pickle.load(open("glmodel_final.pickle", "rb", -1))
with open("model_final_comp.pickle", 'rb') as input :
    cst = pickle.load(input)

@app.route('/index')
def index():
    return flask.render_template('deploy.html')

@app.route('/predict', methods = ['POST'])
def predict():
    datapoint = request.form.to_dict()
    review_text = clean_text(datapoint['personal_story'])
    pred = function_one([review_text])
    if pred == 0 :
        prediction = "Doesn't belong to any class"
    if pred == 1 :
        prediction = "Commenting"
    if pred == 2 :
        prediction = "Ogling"
    if pred == 3 :
        prediction =  "Touching"
    if pred == 4 :
        prediction = "Commenting and Ogling"
    if pred == 5 :
        prediction = "Commenting and Touching"
    if pred == 6 :
        prediction = "Touching and Ogling"
    if pred == 7 :
        prediction = "Commenting, Touching and Ogling"
    return render_template('deploy.html', prediction_text = "Predicted class is {}".format(prediction))

if __name__ == '__main__':
    app.run()