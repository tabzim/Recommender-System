# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 11:42:25 2017

@author: humasamin
"""

from nltk.tokenize import RegexpTokenizer
#from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import logging
from nltk.stem.wordnet import WordNetLemmatizer

lmtzr = WordNetLemmatizer()


#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
#en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# create sample documents



doc_set = []
for i in range(1, 101):
   
    file_object = open(
        "C:\\Training set\\ip" + str(i) + ".txt", "r")
    doc_a = file_object.read()
    doc_set.append(doc_a)

for i in range(1, 112):
   
    file_object = open(
        "C:\\Training set\\Networks" + str(i) + ".txt",
        "r")
    doc_a = file_object.read()
    doc_set.append(doc_a)

for i in range(1, 302):
   
    file_object = open(
        "C:\\Training set\\nips" + str(i) + ".txt",
        "r")
    doc_a = file_object.read()
    doc_set.append(doc_a)

for i in range(1, 71):# use 1-71
   
    file_object = open(
        "C:\\Training set\\KDD" + str(i) + ".txt", "r")
    doc_a = file_object.read()
    doc_set.append(doc_a)
 #for test set   
for i in range(100, 130):
    
    file_object = open(
        "C:\\Training set\\KDD" + str(i) + ".txt", "r")
    doc_a = file_object.read()
    doc_set.append(doc_a)

## Cross Validation Set
#for i in range(71, 101):
#    
#    file_object = open(
#        "C:\\Cross Validation set\\ip" + str(
#            i) + ".txt", "r")
#    doc_a = file_object.read()
#    doc_set.append(doc_a)
#
#for i in range(71, 112):
#    
#    file_object = open(
#        "C:\\Cross Validation set\\Networks" + str(
#            i) + ".txt", "r")
#    doc_a = file_object.read()
#    doc_set.append(doc_a)
#
#for i in range(211, 302):
#    
#    file_object = open(
#        "C:\\Cross Validation set\\nips" + str(
#            i) + ".txt", "r")
#    doc_a = file_object.read()
#    doc_set.append(doc_a)
#
#for i in range(100, 130):
#    
#    file_object = open(
#        "C:\\Cross Validation set\\kdd" + str(
#            i) + ".txt", "r")
#    doc_a = file_object.read()
#    doc_set.append(doc_a)

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
#print(dictionary)
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
#print(corpus)

import os, re

data_dir = 'C:\\Training set Authors'

# Get all document texts and their corresponding IDs.
docs = []
doc_ids = []
id=0
author2doc=dict()
files = os.listdir(data_dir)  # List of filenames.
for filen in files:
    # Get document ID.
    #id=id+1
    #doc_ids.append(id)
    #print(doc_ids)



    # Read document text.
    # Note: ignoring characters that cause encoding errors.
    with open(data_dir + '/' + filen, errors='ignore', encoding='utf-8') as fid:
        txt = fid.read()
        # Replace any whitespace (newline, tabs, etc.) by a single space.
        txt = re.sub('\s', ' ', txt)

        contents = txt.split(",")
        #print(contents)
        for authorname in contents:
            #print(authorname)
            if not author2doc.get(authorname):
                author2doc[authorname]=[id]

            else:
                author2doc[authorname].append(id)
    doc_ids.append(id)
    id = id + 1

    # Replace any whitespace (newline, tabs, etc.) by a single space.
   # txt = re.sub('\s', ' ', txt)


#############################################################
#############Cross Validation set authors
#
#data_dircv = 'C:\\Cross Validation set Authors'
#
## Get all document texts and their corresponding IDs.
#docscv = []
#doc_idscv = []
#idcv=0
#author2doccv=dict()
#filescv = os.listdir(data_dircv)  # List of filenames.
#for filen in filescv:
#    # Get document ID.
#    #id=id+1
#    #doc_ids.append(id)
#    #print(doc_ids)
#
#
#
#    # Read document text.
#    # Note: ignoring characters that cause encoding errors.
#    with open(data_dircv + '/' + filen, errors='ignore', encoding='utf-8') as fidcv:
#        txt = fidcv.read()
#        # Replace any whitespace (newline, tabs, etc.) by a single space.
#        txt = re.sub('\s', ' ', txt)
#
#        contents = txt.split(",")
#        #print(contents)
#        for authorname in contents:
#            #print(authorname)
#            if not author2doccv.get(authorname):
#                author2doccv[authorname]=[idcv]
#
#            else:
#                author2doccv[authorname].append(idcv)
#    doc_idscv.append(idcv)
#    idcv = idcv + 1
#
#     #Replace any whitespace (newline, tabs, etc.) by a single space.
#    #txt = re.sub('\s', ' ', txt)
#
##print(author2doccv)
##print(len(author2doccv))
#

#
###########Test Set##############################
# Folder containing all Test papers
data_dirtest = 'C:\\Test set'  # Set this path to the data on your machine.

# Get all document texts and their corresponding IDs.
docstest = []
doc_idstest = []
idtest=0
filestest = os.listdir(data_dirtest)  # List of filenames.
for filen in filestest:
    # Get document ID.
    idtest = idtest + 1
    doc_idstest.append(idtest)

    # Read document text.
    # Note: ignoring characters that cause encoding errors.
    with open(data_dirtest + '/' + filen, errors='ignore', encoding='utf-8') as fid1:
         txt = fid1.read()

         # Replace any whitespace (newline, tabs, etc.) by a single space.
         txt = re.sub('\s', ' ', txt)

         docstest.append(txt)




# list for tokenized documents in loop
textstest = []

# loop through document list
for i in docstest:
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
    #textstest.append(stemmed_tokens)
    textstest.append(lemmatized_tokens)
    


#print(textstest)






# Create a dictionary representation of the documents, and filter out frequent and rare words.


dictionarytest = corpora.Dictionary(textstest)
dictionary.merge_with(dictionarytest)

# Vectorize data.

# Bag-of-words representation of the documents.
corpus2 = [dictionarytest.doc2bow(t) for t in textstest]
#print(corpus2)


#Creating Author document dictionary
import os, re

data_dir1test = 'C:\\Test set'




# Get all document texts and their corresponding IDs.
docs12 = []
doc_ids12 = []
id12=0
author2doctest=dict()
files12 = os.listdir(data_dir1test)  # List of filenames.
for filen in files12:

    contents=filen.split('.')
    aname=contents[0]
    if not author2doc.get(aname):
        author2doctest[aname]=[id12]

    else:
        author2doctest[aname].append(id12)
    doc_idstest.append(id12)
    id12 = id12 + 1

    # Replace any whitespace (newline, tabs, etc.) by a single space.
   # txt = re.sub('\s', ' ', txt)




atmodel = gensim.models.AuthorTopicModel(corpus[0:612], num_topics=5, id2word=dictionary,author2doc=author2doc, passes=1000,eval_every=None,iterations=2000,alpha=1.0,eta=2.0)
#corpustest=corpus[420:612]
corpustest=corpus2
atmodel.update(corpustest,author2doc=author2doctest,passes=10,iterations=20)
print(atmodel.print_topics(num_topics=5, num_words=15))

#print('Number of authors: %d' % len(author2doctest))
print('Number of authors cross validation: %d' % len(author2doctest))
print('Number of unique tokens: %d' % len(dictionary))
#print('Number of unique tokens in test set: %d' % len(dictionarytest))
print('Number of documents: %d' % len(corpustest))

ll=atmodel.log_perplexity(corpustest,chunk_doc_idx=doc_idstest)
print("likelihood")
print(ll)
print("perplexity")
print(2**(-ll))

####checking 
#for a in atmodel.id2author.values():
#    print(a)
    
    

####################################################################
#######Print Topic distributions for a student
#wat=atmodel.get_author_topics('WajahatulAmin')
#print('Topic Distributions of Student')
#print(wat)


##################################################################
###########Print List of Top Authors##############################
#vec=list()
#topic1list=list()
#topic2list=list()
#topic3list=list()
#topic4list=list()
#topic5list=list()
#authorslist=list()
#
#for f in filestest:
#    contents=f.split('.')
#    aname=contents[0]
#    vec.append(atmodel.get_author_topics(aname))
#    authorslist.append(aname)
#    
##for a in atmodel.id2author.values():
##    vec.append(atmodel.get_author_topics(a))
##    authorslist.append(a)
#
#i=0
#for item in vec:
#        sizeitem=len(item)
#        if sizeitem>=1:
#            sublist1 = list()
#            sublist1.append(authorslist[i])
#            sublist1.append(item[0][1])
#            topic1list.append(sublist1)
#
#        if sizeitem>=2:
#            sublist2=list()
#            sublist2.append(authorslist[i])
#            sublist2.append(item[1][1])
#            topic2list.append(sublist2)
#
#        if sizeitem>=3:
#            sublist3 = list()
#            sublist3.append(authorslist[i])
#            sublist3.append(item[2][1])
#            topic3list.append(sublist3)
#
#        if sizeitem>=4:
#            sublist4 = list()
#            sublist4.append(authorslist[i])
#            sublist4.append(item[3][1])
#            topic4list.append(sublist4)
#
#        if sizeitem>=5:
#            sublist5 = list()
#            sublist5.append(authorslist[i])
#            sublist5.append(item[4][1])
#            topic5list.append(sublist5)
#
#
#        i = i + 1
#
#
#
#from operator import itemgetter
#topic1list.sort(key=itemgetter(1),reverse=True)
#topic2list.sort(key=itemgetter(1),reverse=True)
#topic3list.sort(key=itemgetter(1),reverse=True)
#topic4list.sort(key=itemgetter(1),reverse=True)
#topic5list.sort(key=itemgetter(1),reverse=True)
#
#print("Topic 0")
#for j in range(0,10):
#    print(topic1list[j])
#print("Topic 1")
#for j in range(0, 10):
#    print(topic2list[j])
#print("Topic 2")
#for j in range(0, 10):
#    print(topic3list[j])
#print("Topic 3")
#for j in range(0, 10):
#    print(topic4list[j])
#print("Topic 4")
#for j in range(0, 10):
#    print(topic5list[j])
#########################################################################################
#########################################################################################



###################################################################################################################
##creating graph and creating topic nodes including words as their properties
#from py2neo import Graph, Node,Relationship
#
#
#g = Graph(password="admin")
#
#tx = g.begin()
#topicnodelist=list()
# #creating topic nodes in graph database
#for f in filestest:
#    contents=f.split('.')
#    aname=contents[0]
#    #print()
#    #print(aname)
#    #authornode = Node("Author",authorname=aname)
#    #tx.create(authornode)
#    
#    for topic in atmodel[aname]:   
#       print('Topic:'+str(topic[0])+'   Probability:'+str(topic[1]))
#     #  atrel = Relationship(authornode,str(topic[1]), mary)
#      # tx.create(atrel)
#       words=''
#       for word, prob in atmodel.show_topic(topic[0]):
#           words += word + ' '
#       print('Words: ' + words)
#       tn = Node("Topic",topicname="Topic "+str(topic[0]), topwords=words)
#       tx.create(tn)
#       topicnodelist.append(tn)
#    #tx.commit()
#    break;
##creating author nodes and relationships in graph database
#for f in filestest:
#    contents=f.split('.')
#    aname=contents[0]
#    print()
#    print(aname)
#    authornode = Node("Author",authorname=aname)
#    tx.create(authornode)
#    
#    for topic in atmodel[aname]:   
#       print('Topic:'+str(topic[0])+'   Probability:'+str(topic[1]))
#       atrel = Relationship(authornode,str(topic[1]), topicnodelist[topic[0]])
#       tx.create(atrel)
#       words=''
#       #for word, prob in atmodel.show_topic(topic[0]):
#          # words += word + ' '
#       #print('Words: ' + words)
#       #tn = Node("Topic",topicname="Topic "+str(topic[0]), topwords=words)
#       #tx.create(tn)
#       #topicnode.append(tn)
#tx.commit()
#    #break;
##################################################################################################################

#####Get Author Topic Probabilities
#for f in filestest:
#    contents=f.split('.')
#    aname=contents[0]
#    print()
#    print(aname)
#    atprob=atmodel.get_author_topics(aname)
#    print(atprob)


from gensim.matutils import hellinger,kullback_leibler,sparse2full
######Print Author Similarity for student using helinger distance
#print("Main Author:________________________")
#print('TayyabaAzim')
#vec1=atmodel.get_author_topics('TayyabaAzim')
#
#print('####################### Hellinger ###################')  
#for f1 in filestest:
#       contents1=f1.split('.')
#       aname2=contents1[0]
#   
#       vec2=atmodel.get_author_topics(aname2)
#       sims = hellinger(vec1, vec2)
#
#       #author_size = len(atmodel.author2doc[aname2])
#       print(aname2, sims)
#print('#######################KL Divergence###################')       
#for f1 in filestest:
#       contents1=f1.split('.')
#       aname2=contents1[0]
#   
#       vec2=atmodel.get_author_topics(aname2)
#       simskl = kullback_leibler(vec1, vec2)
#
#       #author_size = len(atmodel.author2doc[aname2])
#       print(aname2, simskl)


#filehl=open("C:\\hellingertayyaba.txt", "a+")
##for author in atmodel.id2author.values():
#print("Main Author:________________________")
#print('TayyabaAzim')
#vec1=atmodel.get_author_topics('TayyabaAzim')
##Writing to a file
#filehl.writelines("Main Author:_______________________\n")
#filehl.writelines('TayyabaAzim'+"\n")
#
#
#for author1 in atmodel.id2author.values():
#        vec2=atmodel.get_author_topics(author1)
#        sims = hellinger(vec1, vec2)
#
#        author_size = len(atmodel.author2doc[author1])
#        print(author1, sims)
#        filehl.writelines(author1+" "+str(sims)+"\n")
#filehl.flush()        
#filehl.close()  
#
#
##KL Divergence
#filehl1=open("C:\\kldivergencetayyaba.txt", "a+")
##for author in atmodel.id2author.values():
#print("Main Author:________________________")
#print('TayyabaAzim')
#vec1=atmodel.get_author_topics('TayyabaAzim')
##Writing to a file
#filehl1.writelines("Main Author:_______________________\n")
#filehl1.writelines('TayyabaAzim'+"\n")
#
#
#for author1 in atmodel.id2author.values():
#        vec2=atmodel.get_author_topics(author1)
#        sims = kullback_leibler(vec1, vec2)
#
#        author_size = len(atmodel.author2doc[author1])
#        print(author1, sims)
#        filehl1.writelines(author1+" "+str(sims)+"\n")
#filehl1.flush()        
#filehl1.close()  
#
#
#########################################################################################################
############FurqanAziz
#filehl2=open("C:\\hellingerfurqan.txt", "a+")
##for author in atmodel.id2author.values():
#print("Main Author:________________________")
#print('FurqanAziz')
#vec1=atmodel.get_author_topics('FurqanAziz')
##Writing to a file
#filehl2.writelines("Main Author:_______________________\n")
#filehl2.writelines('FurqanAziz'+"\n")
#
#
#for author1 in atmodel.id2author.values():
#        vec2=atmodel.get_author_topics(author1)
#        sims = hellinger(vec1, vec2)
#
#        author_size = len(atmodel.author2doc[author1])
#        print(author1, sims)
#        filehl2.writelines(author1+" "+str(sims)+"\n")
#filehl2.flush()        
#filehl2.close()  
#
#
##KL Divergence
#filehl3=open("C:\\kldivergencefurqan.txt", "a+")
##for author in atmodel.id2author.values():
#print("Main Author:________________________")
#print('FurqanAziz')
#vec1=atmodel.get_author_topics('FurqanAziz')
##Writing to a file
#filehl3.writelines("Main Author:_______________________\n")
#filehl3.writelines('FurqanAziz'+"\n")
#
#
#for author1 in atmodel.id2author.values():
#        vec2=atmodel.get_author_topics(author1)
#        sims = kullback_leibler(vec1, vec2)
#
#        author_size = len(atmodel.author2doc[author1])
#        print(author1, sims)
#        filehl3.writelines(author1+" "+str(sims)+"\n")
#filehl3.flush()        
#filehl3.close()  
#
#
########################################################################################################
#
#########################################################################################################
############ZuhaibAshfaqKhan
#filehl22=open("C:\\hellingerZuhaibAshfaqKhan.txt", "a+")
##for author in atmodel.id2author.values():
#print("Main Author:________________________")
#print('ZuhaibAshfaqKhan')
#vec1=atmodel.get_author_topics('ZuhaibAshfaqKhan')
##Writing to a file
#filehl22.writelines("Main Author:_______________________\n")
#filehl22.writelines('ZuhaibAshfaqKhan'+"\n")
#
#
#for author1 in atmodel.id2author.values():
#        vec2=atmodel.get_author_topics(author1)
#        sims = hellinger(vec1, vec2)
#
#        author_size = len(atmodel.author2doc[author1])
#        print(author1, sims)
#        filehl22.writelines(author1+" "+str(sims)+"\n")
#filehl22.flush()        
#filehl22.close()  
#
#
##KL Divergence
#filehl23=open("C:\\kldivergenceZuhaibAshfaqKhan.txt", "a+")
##for author in atmodel.id2author.values():
#print("Main Author:________________________")
#print('ZuhaibAshfaqKhan')
#vec1=atmodel.get_author_topics('ZuhaibAshfaqKhan')
##Writing to a file
#filehl23.writelines("Main Author:_______________________\n")
#filehl23.writelines('ZuhaibAshfaqKhan'+"\n")
#
#
#for author1 in atmodel.id2author.values():
#        vec2=atmodel.get_author_topics(author1)
#        sims = kullback_leibler(vec1, vec2)
#
#        author_size = len(atmodel.author2doc[author1])
#        print(author1, sims)
#        filehl23.writelines(author1+" "+str(sims)+"\n")
#filehl23.flush()        
#filehl23.close()  
#
#
########################################################################################################
#
#
#########################################################################################################
############ShariqHussain
#filehl223=open("C:\\hellingerShariqHussain.txt", "a+")
##for author in atmodel.id2author.values():
#print("Main Author:________________________")
#print('ShariqHussain')
#vec1=atmodel.get_author_topics('ShariqHussain')
##Writing to a file
#filehl223.writelines("Main Author:_______________________\n")
#filehl223.writelines('ShariqHussain'+"\n")
#
#
#for author1 in atmodel.id2author.values():
#        vec2=atmodel.get_author_topics(author1)
#        sims = hellinger(vec1, vec2)
#
#        author_size = len(atmodel.author2doc[author1])
#        print(author1, sims)
#        filehl223.writelines(author1+" "+str(sims)+"\n")
#filehl223.flush()        
#filehl223.close()  
#
#
##KL Divergence
#filehl233=open("C:\\kldivergenceShariqHussain.txt", "a+")
##for author in atmodel.id2author.values():
#print("Main Author:________________________")
#print('ShariqHussain')
#vec1=atmodel.get_author_topics('ShariqHussain')
##Writing to a file
#filehl233.writelines("Main Author:_______________________\n")
#filehl233.writelines('ShariqHussain'+"\n")
#
#
#for author1 in atmodel.id2author.values():
#        vec2=atmodel.get_author_topics(author1)
#        sims = kullback_leibler(vec1, vec2)
#
#        author_size = len(atmodel.author2doc[author1])
#        print(author1, sims)
#        filehl233.writelines(author1+" "+str(sims)+"\n")
#filehl233.flush()        
#filehl233.close()  
#
#
########################################################################################################
#
#
#
#########################################################################################################
############AbdulNasirKhan
#filehl2233=open("C:\\hellingerAbdulNasirKhan.txt", "a+")
##for author in atmodel.id2author.values():
#print("Main Author:________________________")
#print('AbdulNasirKhan')
#vec1=atmodel.get_author_topics('AbdulNasirKhan')
##Writing to a file
#filehl2233.writelines("Main Author:_______________________\n")
#filehl2233.writelines('AbdulNasirKhan'+"\n")
#
#
#for author1 in atmodel.id2author.values():
#        vec2=atmodel.get_author_topics(author1)
#        sims = hellinger(vec1, vec2)
#
#        author_size = len(atmodel.author2doc[author1])
#        print(author1, sims)
#        filehl2233.writelines(author1+" "+str(sims)+"\n")
#filehl2233.flush()        
#filehl2233.close()  
#
#
##KL Divergence
#filehl2333=open("C:\\kldivergenceAbdulNasirKhan.txt", "a+")
##for author in atmodel.id2author.values():
#print("Main Author:________________________")
#print('AbdulNasirKhan')
#vec1=atmodel.get_author_topics('AbdulNasirKhan')
##Writing to a file
#filehl2333.writelines("Main Author:_______________________\n")
#filehl2333.writelines('AbdulNasirKhan'+"\n")
#
#
#for author1 in atmodel.id2author.values():
#        vec2=atmodel.get_author_topics(author1)
#        sims = kullback_leibler(vec1, vec2)
#
#        author_size = len(atmodel.author2doc[author1])
#        print(author1, sims)
#        filehl2333.writelines(author1+" "+str(sims)+"\n")
#filehl2333.flush()        
#filehl2333.close()  


#######################################################################################################








#filehl=open("C:\\hellingertayyaba.txt", "a+")
#filehl=open("C:\\KLDivergencetayyaba.txt", "a+")


#for f in filestest:
#    contents=f.split('.')
#    aname=contents[0]
#    print("Main Author:________________________\n")
#    print(aname)
#    ##Writing to a file
#    filehl.writelines("Main Author:_______________________\n")
#    filehl.writelines(aname+"\n")
#    
#    vec1=atmodel.get_author_topics(aname)
#    
#    for f1 in filestest:
#        contents1=f1.split('.')
#        aname2=contents1[0]
#    
#        vec2=atmodel.get_author_topics(aname2)
#        sims = hellinger(vec1, vec2)
#
#        #author_size = len(atmodel.author2doc[aname2])
#        print(aname2, sims)
#        filehl.writelines(aname2+" "+str(sims)+"\n")
#        
#filehl.flush()        
#filehl.close()          

#KL Divergence
##for f in filestest:
#for aname in atmodel.id2author.values():
#    #contents=f.split('.')
#    #aname=contents[0]
#    print("Main Author:________________________\n")
#    print(aname)
#    ##Writing to a file
#    filehl.writelines("Main Author:_______________________\n")
#    filehl.writelines(aname+"\n")
#    
#    vec1=atmodel.get_author_topics(aname)
#    
#    for aname2 in atmodel.id2author.values():
#    #for f1 in filestest:
#        #contents1=f1.split('.')
#        #aname2=contents1[0]
#    
#        vec2=atmodel.get_author_topics(aname2)
#        sims = kullback_leibler(vec1, vec2)
#
#        #author_size = len(atmodel.author2doc[aname2])
#        print(aname2, sims)
#        filehl.writelines(aname2+" "+str(sims)+"\n")
#        
#filehl.flush()        
#filehl.close() 


