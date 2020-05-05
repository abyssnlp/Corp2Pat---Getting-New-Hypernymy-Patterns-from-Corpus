import nltk
import itertools
import re
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import spacy
nlp=spacy.load('en_core_web_sm')

# Gold hypers
import pandas as pd
hypers=pd.read_table('/home/srawat/Documents/Corp2Pat/wbless/wbless.tsv')

hypers=hypers[hypers.relation=='hyper']
# pd.value_counts(hypers.label)
hypos=list(hypers.word1)
hypernyms=list(hypers.word2)


def alt_alt_alpha(sentence):
    new_sent=[]
    words=sentence.split()
    words=list(itertools.chain.from_iterable([w.split(',') for w in words]))
    words=list(itertools.chain.from_iterable([w.split('-') for w in words]))
    for word in words:
        new_sent.append(''.join(w for w in word if w.isalpha()))
    return re.sub('\s\s+',r' ',' '.join(new_sent).strip())

# # test
# doc=nlp(sent)
# deps=[]
# edges=[]
# for token in doc:
#         for child in token.children:
#             edges.append(('{0}'.format(token.lower_),'{0}'.format(child.lower_)))
# graph=nx.Graph(edges)
# nx.draw(graph,with_labels=True)
# path=nx.shortest_path(graph,source='vehicles',target='car')
# for word in path:
#     for token in doc:
#         if token.text.lower()==word:
#             deps.append(token.dep_)
#         else:
#             pass
# deps='-'.join(deps)        

#! Check if there is no path, array might pop an IndexError, Check on tiny/test corpus
def extract_dep_path(sentence,word1,word2):
    lemmatizer=nltk.stem.WordNetLemmatizer()
    words=nltk.word_tokenize(sentence.lower())
    words=[lemmatizer.lemmatize(word) for word in words]
    sentence=' '.join(words)
    doc=nlp(sentence)
    edges=[]
    deps=[]
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_),'{0}'.format(child.lower_)))
    graph=nx.Graph(edges)
    try:
        path=nx.shortest_path(graph,source=word1,target=word2)
        for word in path:
            for token in doc:
                if token.text.lower()==word:
                    deps.append(token.dep_)
                else:
                    pass
    except nx.NetworkXNoPath:
        pass
    return '-'.join(deps)

            
def extract_patterns(corpus,hypos,hypernyms):
    lemmatizer=nltk.stem.WordNetLemmatizer()
    hypos=[lemmatizer.lemmatize(word.lower()) for word in hypos]
    hypernyms=[lemmatizer.lemmatize(word.lower()) for word in hypernyms]
    extractions=[]
    with open(corpus,'r') as f:
        for line in f:
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                sentence=alt_alt_alpha(sentence).lower()
                words=nltk.word_tokenize(sentence)
                words=[lemmatizer.lemmatize(word) for word in words]
                for w1,w2 in zip(hypos,hypernyms):
                    if w1 in words and w2 in words:
                        dep_path=extract_dep_path(sentence,w1,w2)
                        extractions.append((w1,w2,dep_path))
    return extractions

extractions=extract_patterns('/home/srawat/Documents/UMBC+Wiki/combined_corpus.txt',hypos,hypernyms)

extractions_df=pd.DataFrame(extractions,columns=['word1','word2','dep'])
extractions_df.to_csv('/home/srawat/Documents/Corp2Pat/wbless_dep_extractions.csv')
# pd.value_counts(extractions_df.dep)
# import os
# os.listdir('/home/srawat/Documents/Corp2Pat/')
# import stanfordnlp
# # StanfordNLP 
# def extract_dep_path_alt(sentence,word1,word2):
#     nlp=stanfordnlp.Pipeline()
#     lemmatizer=nltk.stem.WordNetLemmatizer()
#     words=nltk.word_tokenize(sentence)
#     words=[lemmatizer.lemmatize(word.lower()) for word in words]
#     sentence=' '.join(words)
#     doc=nlp(sentence)
#     edges=[]
#     deps=[]
#     for sentence in doc.sentences:
#         for word in sentence.words:
#             edges.append(('{0}'.format(word.text.lower(),'{0}'.format(sentence.words[int(word.governor)-1].text))))



# # test stanfordnlp parent-sister node extraction
# nlp=stanfordnlp.Pipeline()
# doc=nlp(sent)
# for sentence in doc.sentences:
#     for word in sentence.words:
#         print(word.text,sentence.words[int(word.governor)-1].text)


# sentence='Vehicles like car and truck are seen on highways.'
# doc=nlp(sentence)
# edges=[]
# # deps=[]
# for sentence in doc.sentences:
#     for word in sentence.words:
#         edges.append(('{0}'.format(word.text.lower()),'{0}'.format(sentence.words[int(word.governor)-1].text)))
# graph=nx.Graph(edges)
# nx.draw(graph,with_labels=True)
# nx.shortest_path(graph,source='Vehicles',target='truck')