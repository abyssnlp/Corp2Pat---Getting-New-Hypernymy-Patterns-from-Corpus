import networkx as nx
import spacy
import nltk
from tqdm import tqdm
import re
from collections import Counter
import itertools
# Sentences to dependency extractions
nlp=spacy.load('en_core_web_sm')
# def sents_to_dep(extraction_file):
#     nlp=spacy.load('en_core_web_sm')
#     with open(extraction_file,'r') as f:
#         for line in tqdm(f):
#             items=line.strip().split('\t')
#             if len(items) < 3:
#                 w1,w2,sent=line.strip().split('\t')
#             else:
#                 w1=items[0]
#                 w2=items[1]
#                 sent=''.join(items[2:])
#             words=sent.split()
#             words=[re.sub(r'[^\w+]+\w+',r'',word) for word in words]
#             sent=' '.join(words)

#             doc=nlp(sent)
#             edges=[]
#             for token in doc:
#                 for child in token.children:
#                     edges.append((token.text.lower(),child.text.lower()))
#             graph=nx.Graph(edges)
#             try:
#                 path=nx.shortest_path(graph,source=w1,target=w2)
#             except nx.exception.NetworkXNoPath:
#                 pass
#             except nx.exception.NodeNotFound:
#                 pass
#             dep=[]
#             for word in path:
#                 for token in doc:
#                     if word==token.text.lower():
#                         dep.append(token.dep_)
#             dep='#'.join(dep)
#             with open('/home/srawat/Documents/Corp2Pat/dep_extractions.txt','a') as g:
#                 g.write(w1+'\t'+w2+'\t'+dep+'\n')

# sents_to_dep('/home/srawat/Documents/Corp2Pat/sent_extractions.txt')


def match_patterns(corpus, dep_patterns):
    new_extractions=[]
    dep_paths=list(dep_patterns.keys())
    with open(corpus,'r') as f:
        for line in tqdm(f):
            sentences=nltk.sent_tokenize(line)
            for sentence in sentences:
                sentence=sentence.lower()
                word_pairs=list(itertools.permutations(nltk.word_tokenize(sentence),2))
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
                    except nx.exception.NetworkXNoPath:
                        pass
                    except nx.exception.NodeNotFound:
                        pass
    return new_extractions

# For a certain number of epochs, repeat cyclic traversal to get optimal pairs
def cyclic_traversal(corpus,epochs):
    epoch=1
    while epoch<=epochs:
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

cyclic_traversal('/home/srawat/Documents/UMBC+Wiki/combined_corpus.txt',2)