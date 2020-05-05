# test corpus read in

with open('test_corpus.txt','r') as f:
    data=f.read()

import nltk


nltk.download('all')

sentences=nltk.sent_tokenize(data)

# Getting gold hypernyms from WBLESS
import pandas as pd
hypers=pd.read_table('wbless/wbless.tsv')

pd.value_counts(hypers.relation).plot(kind='bar')

pd.value_counts(hypers.label)
# select only hypernym relations 
hypers=hypers[hypers.relation=='hyper'][['word1','word2']]


# Extract sentences from the corpus that have both the terms of the pair

# Approach:
# split corpus into fixed sentences
# map each sentence to finding the sentences with matched pairs
# function for extracting matches sentences into 3 lists 
# write lists to drive
# later combine to form dataframe

# split using bash script

from nltk.stem import WordNetLemmatizer
import string

# add after exec
lemmatizer=WordNetLemmatizer()
punkt=string.punctuation
def preprocess(sentence):
    words=sentence.split()
    words=[word for word in words if word not in punkt]
    words=[lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)
#############################

def extract_sent_pairs(filename):
    data=open('data/'+filename).read()
    sentences=ntlk.sent_tokenize(data)
    word1=hypers.word1
    word2=hypers.word2
    w1=[]
    w2=[]
    sent=[]
    for sentence in sentences:
        if word1 in sentence and word2 in sentence:
            w1.append(word1)
            w2.append(word2)
            sent.append(sentence)
    return w1,w2,sent


import multiprocessing as mp
import os
datafiles=os.listdir('data/')
cpu=mp.cpu_count()
pool=mp.Pool(processes=cpu)
w1,w2,sent=pool.map(extract_sent_pairs,datafiles)
