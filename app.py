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
        pred = clf.predict(i)
        return pred

dictionary = pickle.load(open("dictionary_final.pickle", "rb", -1))
tfidf_words = pickle.load(open("tfwords_final.pickle", "rb", -1))
glove_words = pickle.load(open("glove_final.pickle", "rb", -1))
model_vec = pickle.load(open("glmodel_final.pickle", "rb", -1))
clf = pickle.load(open("model_final_xg.pickle", 'rb',-1))

@app.route('/index')
def index():
    return flask.render_template('deploy.html')
@app.route('/')
def home():
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