# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:13:36 2017

@author: humasamin
"""

from nltk.tokenize import RegexpTokenizer
#from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import logging
from gensim.matutils import kullback_leibler,hellinger
from nltk.stem.wordnet import WordNetLemmatizer

lmtzr = WordNetLemmatizer()


#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
#en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# create sample documents



doc_set=[]
for i in range(1,101):
    
    file_object = open("C:\\Training set\\ip" + str(i) + ".txt", "r")
    doc_a = file_object.read()
    doc_set.append(doc_a)

for i in range(1,112):
   
    file_object = open("C:\\Training set\\Networks" + str(i) + ".txt", "r")
    doc_a = file_object.read()
    doc_set.append(doc_a)


for i in range(1,302):
    
    file_object = open("C:\\Training set\\nips" + str(i) + ".txt", "r")
    doc_a = file_object.read()
    doc_set.append(doc_a)

for i in range(1,71):
    
    file_object = open("C:\\Training set\\KDD" + str(i) + ".txt", "r")
    doc_a = file_object.read()
    doc_set.append(doc_a)

for i in range(100,130):
    
    file_object = open("C:\\Training set\\KDD" + str(i) + ".txt", "r")
    doc_a = file_object.read()
    doc_set.append(doc_a)
    

##Cross Validation Set
#for i in range(71,101):
#    
#    file_object = open("C:\\Cross Validation set\\ip" + str(i) + ".txt", "r")
#    doc_a = file_object.read()
#    doc_set.append(doc_a)
#
#for i in range(71,112):
#    
#    file_object = open("C:\\Cross Validation set\\Networks" + str(i) + ".txt", "r")
#    doc_a = file_object.read()
#    doc_set.append(doc_a)
#
#
#for i in range(211,302):
#    
#    file_object = open("C:\\Cross Validation set\\nips" + str(i) + ".txt", "r")
#    doc_a = file_object.read()
#    doc_set.append(doc_a)
#
#for i in range(100,130):
#    
#    file_object = open("C:\\Cross Validation set\\kdd" + str(i) + ".txt", "r")
#    doc_a = file_object.read()
#    doc_set.append(doc_a)


#Test set
import os,re

data_dir = 'C:\\Test set'
files = os.listdir(data_dir)
for filen in files:
    
    # Read document text.
    # Note: ignoring characters that cause encoding errors.
    with open(data_dir + '/' + filen, errors='ignore', encoding='utf-8') as fid:
        txt = fid.read()
        # Replace any whitespace (newline, tabs, etc.) by a single space.
        txt = re.sub('\s', ' ', txt)

        doc_set.append(txt)
    

#print(doc_set)


# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    #stopped_tokens = [i for i in tokens if not i in en_stop]
    stop_words=set(stopwords.words("english"))
    stopped_tokens = [i for i in tokens if not i in stop_words]

    # stem tokens
    #stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    lemmatized_tokens=[lmtzr.lemmatize(i) for i in stemmed_tokens]

    # add tokens to list
    #texts.append(stemmed_tokens)
    texts.append(lemmatized_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]


# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus[0:420], num_topics=5, id2word=dictionary, passes=1000,eval_every=None,iterations=1000,alpha=1.0,eta=0.01)

#corpus2=corpus[612:698]
corpus2=corpus[420:612]
ldamodel.update(corpus2,passes=1000,iterations=1000)
print(ldamodel.print_topics(num_topics=5, num_words=15))
ll=ldamodel.log_perplexity(corpus2,total_docs=None)
print("likelihood")
print(ll)
print("perplexity")
print(2**(-ll))
print("words in dictionary")
print(len(dictionary))
print(len(corpus2))





