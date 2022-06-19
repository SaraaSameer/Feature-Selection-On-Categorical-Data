#######################  Importing All Necessary Libraries ################################## 
from argparse import _VersionAction
import pandas as pd
import re
import pickle
import copy
import os
from bs4 import BeautifulSoup
from collections import defaultdict


from sklearn.feature_extraction.text import TfidfVectorizer            
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


"""" One time function to create csv from parsed data, 
    file.csv is already loaded, no need to call this function """
def parse_data():
    text = []
    tag = []
    directory ='course-cotrain-data/fulltext/course'
    for filename in os.listdir(directory):
        fname = os.path.join(directory,filename)
        with open(fname, 'r') as f:
            soup = BeautifulSoup(f.read(),'html.parser')
            text.append(soup.text)
            tag.append('course')    # Adding target variable

    directory ='course-cotrain-data/fulltext/non-course'
    for filename in os.listdir(directory):
        fname = os.path.join(directory,filename)
        with open(fname, 'r') as f:
            soup = BeautifulSoup(f.read(),'html.parser')
            text.append(soup.text)
            tag.append('non-course')   # Adding target variable

    df = pd.DataFrame({'WebText': text, 'class': tag})
    df.to_csv('file.csv', index=False)

"""" parse_data()   No need to call this function again, it takes a lot of time to parse data. 
                    Hence is loaded in csv file """

# The feature selection code is in seperate file. features are saved in a .txt file for quick access.
# Here features are directly read from file, to see the working please refer features_selection.py file.

topnouns = set()    #Using sets structure to remove any duplicates in the stored list
def read_topnoun_txt():  
    file1 = open('features/top_nouns.txt', 'r')
    Lines = file1.readlines()
    for line in Lines:
        line = line.strip()   # remove \n
        topnouns.add(line)

tfidf_100 = set()
def read_tfidf_txt():  
    file1 = open('features/tfidf_top100.txt', 'r')
    Lines = file1.readlines()
    for line in Lines:
        line = line.strip()   # remove \n
        tfidf_100.add(line)

lexical_chains = set()
def read_lexchains_txt():  
    file1 = open('features/lexchains.txt', 'r')
    Lines = file1.readlines()
    for line in Lines:
        line = line.strip()   # remove \n
        lexical_chains.add(line)

topic_terms = set()
def read_topicterms_txt():
    file1 = open('features/topicmodeling_lda.txt', 'r')
    Lines = file1.readlines()
    for line in Lines:
        line = line.strip()   # remove \n
        topic_terms.add(line)

read_topnoun_txt()
read_tfidf_txt()
read_lexchains_txt()
read_topicterms_txt()
df = pd.read_csv('file.csv')   #Reading csv into dataframe

# Part 4: Merge all technqiues
# Excluded LDA from intersection because good accuarcy was not achieved.
features_intersect = topnouns.intersection(tfidf_100.intersection(lexical_chains))   ## 18 features in total

def feature_engineering():
    vect = TfidfVectorizer()
    X=vect.fit(features_intersect)           #Fitting features to make the vocabulary
    X= vect.transform(df['WebText'])         #Transforming doc matrix to that vocabulary, extra features that are not present in vocabulary will be discarded
    return X, vect

X, vect = feature_engineering()


## Part 4: And then use it for NB Space
def NB_Model():
    Y = df['class']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, stratify = Y, random_state = 42)
    model = MultinomialNB() #Suitable for large docs
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    return predictions, Y_test, model

predictions, Y_test, model = NB_Model()

################ Saving Model and Vectorizer #########################
pickle.dump(model, open('models\model.pickle', 'wb'))
pickle.dump(vect, open("models\\vect.pickle", "wb"))



############################### Validation Metrics ########################################
def accuracy_scr():
    score = accuracy_score(Y_test, predictions) *100
    return score

def prec_scr():
    score = precision_score(Y_test, predictions, average=None)
    return score

def recall_scr():
    score = recall_score(Y_test, predictions, average=None)
    return score

def f1_scr():
    score = f1_score(Y_test, predictions, average=None)
    return score

# This report contains precision, recall, F1 measure and suppport
def classify_report():
    print(classification_report(Y_test, predictions))


# Validation metrices
accuracy = accuracy_scr()
precision = prec_scr()
recall = recall_scr()
f1_measure = f1_scr()
classify_report()

print(f'The accuracy % is: {round(accuracy,2)}%')
print(f'Precision of Class["Course"] is {round(precision[0]*100,2)}%')
print(f'Precision of Class["NonCourse"] is {round(precision[1]*100,2)}%')
print(f'Recall of Class["Course"] is {round(recall[0]*100,2)}%')
print(f'Recall of Class["NonCourse"] is {round(recall[1]*100,2)}%')
print(f'F1_Measure of Class["Course"] is {round(f1_measure[0]*100,2)}%')
print(f'F1_Measure of Class["NonCourse"] is {round(f1_measure[1]*100,2)}%')


####################################### Input Query Processing #######################################
############### This is an additional thing, query prediction is not the requirement of our assignment
############### Due to time constraints,I have taken some help from github for this additional piece of code
def Predict_Query(query):
    tfidf_vectorizer = pickle.load(open('models\\vect.pickle','rb'))
    corpus = defaultdict(None, copy.deepcopy(tfidf_vectorizer.vocabulary_))
    corpus.default_factory = corpus.__len__
    m = pickle.load(open('models\model.pickle','rb'))
    tfidf_transformer_query = TfidfVectorizer()
    tfidf_transformer_query.fit_transform(query)
    for word in tfidf_transformer_query.vocabulary_.keys():
        if word in tfidf_vectorizer.vocabulary_:
            corpus[word]
    tfidf_transformer_query = TfidfVectorizer(vocabulary=corpus)
    query_tfidf = tfidf_transformer_query.fit_transform(query)
    return m.predict(query_tfidf)



