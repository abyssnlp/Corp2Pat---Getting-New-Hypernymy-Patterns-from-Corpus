# Corpus to patterns script

import nltk
import itertools
import re
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import spacy
import pandas as pd
from collections import Counter
import argparse
import multiprocessing as mp
from tqdm import tqdm
nlp=spacy.load('en_core_web_sm')

# Agnostic to the gold standard used
# Change labeling for datasets
# Current state: WBLESS
def prepare_gold_standard(path_to_gold):
    hyp_df=pd.read_table(path_to_gold)
    hyp_df=hyp_df[hyp_df.relation=='hyper']
    hypos=list(hyp_df.word1)
    hypernyms=list(hyp_df.word2)
    return hypos,hypernyms


#! Write Pre-processsing algorithm
# First run without pre-processing

# Lazy loader for corpus
def corpusreader(dir):
    with open(dir,'r') as f:
        for line in f:
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                yield sentence.lower()

# Extract hyponym, hypernym and sentences from the text
def extract_words_sents(corpus,hypos,hypernyms):
    sentence=corpusreader(corpus)
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

# From the above sentence extractions, get the dependency paths between pairs
def sents_to_dep(extraction_file):
    nlp=spacy.load('en_core_web_sm')
    with open(extraction_file,'r') as f:
        for line in tqdm(f):
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

# Match patterns from the corpus
def match_patterns(corpus, dep_patterns):
    new_extractions=[]
    dep_paths=list(dep_patterns.keys())
    with open(corpus,'r') as f:
        for line in tqdm(f):
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

# For a certain number of epochs, repeat cyclic traversal to get optimal pairs
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
    with open('/home/srawat/Documents/Corp2Pat/corp2pat_main.txt','w') as g:
        for w1,w2 in new_pairs:
            g.write(w1+'\t'+w2+'\n')
    return new_pairs        

# Execute all together
def execute(corpus,path_to_gold,epochs):
    hypos,hypernyms=prepare_gold_standard(path_to_gold)
    return cyclic_traversal(corpus,hypos,hypernyms,epochs)

# CLI argument parser
parser=argparse.ArgumentParser()

parser.add_argument('--corpus','-C',help='Path to the main corpus')
parser.add_argument('--gold','-G',help='Path to the gold hypernym file')
parser.add_argument('--epochs','-E',help='Number of Epochs to run the cyclic traversal')

args=parser.parse_args()

if args.corpus and args.gold and args.epochs:
    execute(args.corpus,args.gold,int(args.epochs))