######################## Preprocessing ########################
import pandas as pd
import nltk
import re
import numpy as np
from collections import defaultdict
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

from sklearn.decomposition import LatentDirichletAllocation  #Topic Mode
from sklearn.feature_extraction.text import TfidfVectorizer   

df = pd.read_csv('file.csv')    # Reading file.csv into a dataframe 
WebText_collection = []              # Storing it for later use of lemmatization
for row in list(df['WebText']):
    WebText_collection.append(row)  
nltk.download('stopwords')
ps = PorterStemmer()

def cleaning(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)   #Dropping all encodings, numbers etc
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

df['WebText'] = df['WebText'].apply(cleaning)

################################# Technique 01 : TF IDF #############################################

def tf_idf():
    vect = TfidfVectorizer(ngram_range=(1,2), binary=True, max_features=100)   # Selecting top 100 features
    X=vect.fit_transform(df['WebText'])
    features = vect.get_feature_names()
    return features

# Already loaded tfidf_top100.txt file in features folder
def write_tfidf_txt(feat):
    with open('features/tfidf_top100.txt', 'w+') as fp:
        for item in feat:
                  fp.write("%s\n" % item)
        fp.close()

write_tfidf_txt(tf_idf())  # Passing features to store in a txt file  

################################# Technique 02(a): Frequent Nouns #######################################

nltk.download('wordnet')
nltk.download('omw-1.4')
    
def cleaning_lemma(content):  
    lemmatizer = WordNetLemmatizer()
    content = re.sub('[^a-zA-Z]',' ',content)   #Dropping all encodings, numbers etc
    content = content.lower()
    content = content.split()
    # Using lemmatization for noun extraction
    content = [lemmatizer.lemmatize(word) for word in content if not word in stopwords.words('english')]
    content = ' '.join(content)
    return content

cleaned_tokens= [cleaning_lemma(doc) for doc in WebText_collection]

nouns = []     #Save nouns of all collection
def find_nouns(text):
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words = [word for word in words if word not in set(stopwords.words('english'))]
        tagged = nltk.pos_tag(words)
        for (word, tag) in tagged:
            if tag == 'NN': # If the word is a  noun
                nouns.append(word)


for text in cleaned_tokens:
     find_nouns(text)

# Already loaded, this file contains all nouns to be used in lexical chains
def write_txt(words):
    with open('features/nouns.txt', 'w+') as fp:
        for item in words:
            fp.write("%s\n" % item)
    fp.close()
write_txt(set(nouns))   #To drop duplicates

def freq50_nouns():
    counts = Counter(nouns)
    sorted_nouns = dict()
    sorted_freq= sorted(counts,key=counts.get, reverse = True)
    for w in sorted_freq:
        sorted_nouns[w] = counts[w]
    freq_nouns_50 = {k: sorted_nouns[k] for k in list(sorted_nouns)[:50]}
    return freq_nouns_50

freq_nouns_50 = freq50_nouns()

# Already loaded and saved in features folder, this contains top 50 frequent nouns
def write_topnouns_txt(nouns_dict):
    with open('features/top_nouns.txt', 'w+') as fp:
            for value in nouns_dict.keys():
                  fp.write("%s\n" % value)
    fp.close()
 
write_topnouns_txt(freq_nouns_50)

################################# Technique02(b): Topic Models(LDA) #############################
def tokenization():
    clean_collection_tokens = [cleaning(doc).split() for doc in list(df['WebText'])]
    tf_idf_vect = TfidfVectorizer(tokenizer=lambda doc:doc, lowercase=False)
    tf_idf_arr = tf_idf_vect.fit_transform(clean_collection_tokens)
    # Now create vocabulary for tf_idf
    vocab_tf_idf = tf_idf_vect.get_feature_names()
    return tf_idf_arr, vocab_tf_idf

tf_idf_arr, vocab_tf_idf = tokenization()

topic_terms =[]
def lda_modeling():
    #n_components means number of topics to be retrieved.
    lda_model = LatentDirichletAllocation(n_components = 18, max_iter = 20, random_state = 20)
    X_topics = lda_model.fit_transform(tf_idf_arr)
    # print(len(X_topics))
    topic_words = lda_model.components_
    n_words = 5   #Retrieve top 5 words from each topic
    for i, topic_dist in enumerate(topic_words):
        sorted_topic_dist = np.argsort(topic_dist)
        topic_words = np.array(vocab_tf_idf)[sorted_topic_dist]
        topic_words = topic_words[:-n_words:-1]
        # print(topic_words)
        for j in topic_words:
            topic_terms.append(j)

lda_modeling()
    
def write_topicterms_txt():
    with open('features/topicmodeling_lda.txt', 'w+') as fp:
        for item in topic_terms:
                  fp.write("%s\n" % item)
        fp.close()

write_topicterms_txt()
  

################################## Technique03: Lexical Chains ###################################

def relation_list(nouns):

    rel = defaultdict(list) 
    for k in range (len(nouns)):   
        temp = []
        # using nouns to define relations between terms
        for syn in wordnet.synsets(nouns[k], pos = wordnet.NOUN):
            for l in syn.lemmas():
                temp.append(l.name())
                if l.antonyms():
                    temp.append(l.antonyms()[0].name())
            for l in syn.hyponyms():
                if l.hyponyms():
                    temp.append(l.hyponyms()[0].name().split('.')[0])
            for l in syn.hypernyms():
                if l.hypernyms():
                    temp.append(l.hypernyms()[0].name().split('.')[0])
        rel[nouns[k]].append(temp)
        
    return rel

nouns = []

# Read nouns from the saved file
def read_noun_txt():  
    file1 = open('features/nouns.txt', 'r')
    Lines = file1.readlines()
    for line in Lines:
        line = line.strip()   # remove \n
        nouns.append(line)

read_noun_txt()
relations = relation_list(nouns)

def lex_chains(nouns, relation_list):
    lex = []
    threshold = 0.6   #Setting a random threshold
    for noun in nouns:
        flag = 0
        for j in range(len(lex)):
            if flag == 0:
                for key in list(lex[j]):
                    if key == noun and flag == 0:
                        lex[j][noun] +=1
                        flag = 1
                    elif key in relation_list[noun][0] and flag == 0:
                        s1 = wordnet.synsets(key, pos = wordnet.NOUN)
                        s2 = wordnet.synsets(noun, pos = wordnet.NOUN)
                        if s1[0].wup_similarity(s2[0]) >= threshold:
                            lex[j][noun] = 1
                            flag = 1
                    elif noun in relation_list[key][0] and flag == 0:
                        s1 = wordnet.synsets(key, pos = wordnet.NOUN)
                        s2 = wordnet.synsets(noun, pos = wordnet.NOUN)
                        if s1[0].wup_similarity(s2[0]) >= threshold:
                            lex[j][noun] = 1
                            flag = 1
        if flag == 0: 
            dic = {}
            dic[noun] = 1
            lex.append(dic)
            flag = 1

    final_chain = []
    # Removing weak lex chains
    while lex:
        result = lex.pop()
        if len(result.keys()) == 1:
            for item in result.values():
                if item != 1: 
                    final_chain.append(result)
        else:
            final_chain.append(result)

    return final_chain

lex_chain = lex_chains(nouns, relations)

# Storing lexchain features in seperate file, already saved
def write_lexchain_txt(lexchains):
     with open('features\lexchains.txt', 'w+') as fp:
        for item in lexchains:
            for value in item.keys():
                  fp.write("%s\n" % value)
        fp.close()

write_lexchain_txt(lex_chain)