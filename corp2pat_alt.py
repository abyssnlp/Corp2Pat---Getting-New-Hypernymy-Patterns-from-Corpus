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

#! TODO: Proper Preprocessing Algorithm

def alt_alt_alpha(sentence):
    new_sent=[]
    words=sentence.split()
    words=list(itertools.chain.from_iterable([w.split(',') for w in words]))
    words=list(itertools.chain.from_iterable([w.split('-') for w in words]))
    return 


def corpusreader(dir):
    with open(dir,'r') as f:
        for line in f:
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                yield alt_alt_alpha(sentence).lower()

def corpusreader_alt(dir):
    with open(dir,'r') as f:
        for line in f:
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                yield sentence.lower()

# # test
# extractions=[]
# for sent in corpusreader('/home/srawat/Documents/UMBC+Wiki/test/test_corpus.txt'):
#     for w1,w2 in zip(hypos,hypernyms):
#         if w1 in sent and w2 in sent:
#             extractions.append(tuple([w1,w2,sent]))


# test
extractions=[]
sentence=corpusreader_alt('/home/srawat/Documents/UMBC+Wiki/test/test_corpus.txt')
while True:
    try:
        sent=next(sentence)
        words=nltk.word_tokenize(sent)
        for w1,w2 in zip(hypos,hypernyms):
            if w1 in words and w2 in words:
                extractions.append(tuple([w1,w2,sent]))
    except StopIteration:
        break

# Write extractions to file on the fly

def extract_words_sents(corpus,hypos,hypernyms):
    sentence=corpusreader_alt(corpus)
    while True:
        try:
            sent=next(sentence)
            words=nltk.word_tokenize(sent)
            for w1,w2 in zip(hypos,hypernyms):
                if w1 in words and w2 in words:
                    with open('/home/srawat/Documents/Corp2Pat/sent_extractions.txt','a') as g:
                        g.write(w1+'\t'+w2+'\t'+sent+'\n')
        except StopIteration:
            break

# Extraction file into dep paths between words

sent=extractions[52][2]
doc=nlp(sent)
edges=[]
for token in doc:
    for child in token.children:
        edges.append((token.text.lower(),child.text.lower()))
graph=nx.Graph(edges)        
nx.draw(graph,with_labels=True)
path=nx.shortest_path(graph,source='countries',target='portugal')
dep=[]
for word in path:
    for token in doc:
        if word==token.text.lower():
            dep.append(token.dep_)

import stanfordnlp
nlp=stanfordnlp.Pipeline()
doc=nlp(sent)
for sentence in doc.sentences:
    for word in sentence.words:
        print(word.text,word.dependency_relation)

sent='Countries like Spain, France and Portugal lie to the south of.'


def sents_to_dep(extraction_file):
    nlp=spacy.load('en_core_web_sm')
    with open(extraction_file,'r') as f:
        for line in f:
            w1,w2,sent=line.strip().split('\t')
            doc=nlp(sent)
            edges=[]
            for token in doc:
                for child in token.children:
                    edges.append((token.text.lower(),child.text.lower()))
            graph=nx.Graph(edges)
            try:
                path=nx.shortest_path(graph,source=w1,target=w2)
            except nx.NetworkXNoPath:
                pass
            dep=[]
            for word in path:
                for token in doc:
                    if word==token.text.lower():
                        dep.append(token.dep_)
            dep='#'.join(dep)
            with open('/home/srawat/Documents/Corp2Pat/dep_extractions.txt','a') as g:
                g.write(w1+'\t'+w2+'\t'+dep+'\n')
            
extract_words_sents('/home/srawat/Documents/UMBC+Wiki/test/test_corpus.txt',hypos,hypernyms)
sents_to_dep('/home/srawat/Documents/Corp2Pat/sent_extractions.txt')


# Count extractions of patterns
deps=[]
with open('/home/srawat/Documents/Corp2Pat/dep_extractions.txt','r') as f:
    for line in f:
        _,_,dep=line.split('\t')
        deps.append(dep.strip())

from collections import Counter
dep_dict=dict(Counter(deps))
dep_dict={k: v for k, v in sorted(dep_dict.items(), reverse=True, key=lambda item: item[1])}

# Check all extractions > 2 and search these patterns from text to get new pairs
# test
from itertools import permutations
edges=[]
depends=[]
sent='Countries like Spain and France are in the south.'
doc=nlp(sent)
for token in doc:
    for child in token.children:
        edges.append((token.text.lower(),child.text.lower()))
graph=nx.Graph(edges)
path=nx.shortest_path(graph,source='france',target='countries')
for word in path:
    for token in doc:
        if token.text.lower()==word:
            depends.append(token.dep_)
sent_dep=[]
word_pairs=[]
for words in list(itertools.permutations(sent.split(),2)):
    word_pairs.append(words)

# for token in doc:
#     for words in list(itertools.permutations(sent.split(),2)):
        
# # extract words using dep path
# paths=list(dep_dict.keys())
# with open('/home/srawat/Documents/UMBC+Wiki/test/tiny_corpus.txt','r') as f:
#     for line in f:
#         sentences=nltk.sent_tokenize(line)
#         for sentence in sentences:
#             for path in paths:
                


def match_patterns(corpus, dep_patterns):
    new_extractions=[]
    dep_paths=list(dep_patterns.keys())
    with open(corpus,'r') as f:
        for line in f:
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                sentence=sentence.lower()
                word_pairs=list(itertools.permutations(nltk.word_tokenize(sent),2))
                doc=nlp(sentence)
                edges=[]
                for token in doc:
                    for child in token.children:
                        edges.append((token.text.lower(),child.text.lower()))
                graph=nx.Graph(edges)
                for w1,w2 in word_pairs:
                    try:
                        deps=[]
                        path=nx.shortest_path(graph,source=w1,target=w2)
                        for word in path:
                            for token in doc:
                                if token.text.lower()==word:
                                    deps.append(token.dep_)
                        for dep_path in dep_paths:
                            dep_path=dep_path.split('#')
                            if dep_path==deps:
                                new_extractions.append((w1,w2))
                    except nx.NetworkXNoPath:
                        pass
    return new_extractions

from collections import Counter

def cyclic_traversal(corpus,hypos,hypernyms,epochs):
    epoch=1
    while epoch<=epochs:
        extract_words_sents(corpus,hypos,hypernyms)
        sents_to_dep('/home/srawat/Documents/Corp2Pat/sent_extractions.txt')
        deps=[]
        with open('/home/srawat/Documents/Corp2Pat/dep_extractions.txt','r') as f:
            for line in f:
                _,_,dep=line.split('\t')
                deps.append(dep.strip())

        dep_dict=dict(Counter(deps))
        dep_dict={k: v for k, v in sorted(dep_dict.items(), reverse=True, key=lambda item: item[1])}
        new_pairs=match_patterns(corpus,dep_dict)
        hypos=[]
        hypers=[]
        for w1,w2 in new_pairs:
            hypos.append(w1)
            hypers.append(w2)
        epoch+=1
    return new_pairs        

# Prepare hypo, hyper pairs from Wordnet, DBPedia, Wikidata and Yago

# test
wn.synset('dog.n.01').hypernyms()[0].name().split('.')[0]

# wordnet
from nltk.corpus import wordnet as wn
hypernyms=[]
hyponyms=[]
for word in wn.words():
    for item in wn.synsets(word)[0].hypernyms():
        hypernyms.append(item.name().split('.')[0])
        hyponyms.append(word)
    for item in wn.synsets(word)[0].hyponyms():
        hypernyms.append(word)
        hyponyms.append(item.name().split('.')[0])
with open('/home/srawat/Documents/Corp2Pat/wordnet_hypernyms.txt','w') as f:
    for hypo,hyper in zip(hyponyms,hypernyms):
        f.write(hypo+'\t'+hyper+'\n')            
#! Clean Wordnet hypernym pairs
# DBpedia hypernyms
# import pandas as pd
# from SPARQLWrapper import SPARQLWrapper,JSON
# sparql=SPARQLWrapper("http://dbpedia.org")
# sparql.setReturnFormat(JSON)
# sparql.setQuery("""PREFIX rdfs <http://www.w3.org/2000/01/rdf-schema#>
# SELECT * WHERE {?s rdfs:subClassOf ?o.
# FILTER regex(?s,"http://dbpedia.org/ontology/")
#  } LIMIT 1000""")

# test=sparql.query()
# test_dict=test.convert()

# pd.DataFrame(sparql.query().convert())
# sparql.query().convert().to_csv('/home/srawat/Documents/Corp2Pat/dbpedia_hypernyms.csv')

# for result in test["results"]["bindings"]:
#     print(result["label"]["value"])

#! Broken API, get from database: https://dbpedia.org/sparql.

# func to get results SPARQL endpoint results
def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

# dbpedia
dbpedia_url='http://dbpedia.org/sparql'
query="""select * where {?s rdfs:subClassOf ?o. 
        NOT EXISTS {FILTER regex(?o,"http://www.w3.org/1999/02/22-rdf-syntax-ns#Property")}}"""
alt_query="""select * where {?s rdf:type ?o.}"""

dbpedia_results=get_results(dbpedia_url,alt_query)
for result in dbpedia_results['results']['bindings']:
    print(result)



# Wikidata API test
import sys
from SPARQLWrapper import SPARQLWrapper, JSON

endpoint_url = "https://query.wikidata.org/sparql"

query = """select * where {?s a ?o.} limit 1000"""

results = get_results(endpoint_url, query)

for result in results["results"]["bindings"]:
    print(result)
