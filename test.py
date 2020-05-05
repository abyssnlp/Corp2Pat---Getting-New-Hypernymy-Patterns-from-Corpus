import nltk
import re
lemmatizer=nltk.stem.WordNetLemmatizer()

# read in and clean corpus
with open('test_corpus.txt','r') as f:
    data=f.read()

def clean_line(file_line):
    sentences=nltk.sent_tokenize(file_line)
    new_sentences=[]
    for sentence in sentences:
        #sentence=re.sub(r'[^a-zA-z\s]',r'',sentence)
        #sentence=re.sub("(?<=[a-z])'(?=[a-z])", "", sentence)
        words=sentence.split()
        words=[word.lower() for word in words]
        #words=[lemmatizer.lemmatize(word) for word in words]
        sentence=' '.join(words)
        new_sentences.append(sentence)
    return new_sentences
    

# test yield 
def get_sentences(filename): 
    with open(filename,'r') as f:
        for line in f:
            yield line

# sentences=get_sentences('test_corpus.txt')
# type(next(sentences))

# Gold hypernyms
import pandas as pd
hypers=pd.read_table('/home/srawat/Documents/Hearst Pattern Analysis/wbless/wbless.tsv')
hypers=hypers[hypers.relation=='hyper']
hypers=hypers.reset_index(drop=True)
hypos=hypers.word1
hypers=hypers.word2

# Extractions from corpus
def extract_sents(filename,hypos,hypers):
    extractions=[]
    with open(filename,'r') as f:
        for line in f:
            sents=clean_line(line)
            for sent in sents:
                words=sent.split()
                for w1,w2 in zip(hypos,hypers):
                    if w1 in words and w2 in words:
                        extractions.append((sent,w1,w2))
        return extractions

# Extractions from corpus without pre-processing
def extract_sents(filename,hypos,hypers):
    extractions=[]
    with open(filename,'r') as f:
        for line in f:
            sents=nltk.sent_tokenize(line)
            for sent in sents:
                words=sent.split()
                for w1,w2 in zip(hypos,hypers):
                    if w1 in words and w2 in words:
                        extractions.append((sent,w1,w2))
        return extractions
test_extractions=extract_sents('test_corpus.txt',hypos,hypers)
 
#   !TODO significantly less extractions without pre-processing

# Check dep paths between pairs
# Format: sent,hypo,hyper
import networkx as nx
import spacy
nlp=spacy.load('en_core_web_sm')

def extract_dep_paths(extractions):
    dep_paths=[]
    for sent,hypo,hyper in extractions:
        doc=nlp(sent)
        edges=[]
        for token in doc:
            for child in token.children:
                edges.append(('{0}'.format(token.lower_),'{0}'.format(child.lower_)))
        graph=nx.Graph(edges)
        try:
            path=nx.shortest_path(graph,source=hypo,target=hyper)
            paths=[]
            for word in path:
                for token in doc:
                    if word==token.text.lower():
                        paths.append(token.text+'-'+token.pos_+'-'+token.dep_)
            paths=' '.join(paths)
            dep_paths.append((paths,hypo,hyper))
        except nx.NetworkXNoPath:
            pass
    return dep_paths

test_dep_paths=extract_dep_paths(test_extractions)

# All dependencies
all_deps=[]
for i in range(len(test_dep_paths)):
    sent=test_dep_paths[i][0]
    deps=[x.split('-')[2] for x in sent.split()]
    all_deps.append(deps)

# unique dependencies
import itertools
deps=list(itertools.chain.from_iterable(all_deps))
deps=list(set(deps))

# Dict vocab for deps
dep_dict={}
for i in range(len(deps)):
    if deps[i] in dep_dict:
        pass
    else:
        dep_dict[deps[i]]=i

# Encode dep paths extracted and freq
enc_all_deps=[]
for dep in all_deps:
    _=[]
    for obj in dep:
        _.append(dep_dict[obj])
    enc_all_deps.append(_)

# 
# more_than_once=[]
# for dep in enc_all_deps:
#     for dep2 in enc_all_deps:
#         if dep==dep2:
#             more_than_once.append(dep)

dep_freq_dict={}
for dep in enc_all_deps:
    if tuple(dep) in dep_freq_dict:
        dep_freq_dict[tuple(dep)]+=1
    else:
        dep_freq_dict[tuple(dep)]=1

# stanford NLP check
# check spacy github repo, for dependency matcher from text

    





# test
test=test_dep_paths[1][0]
[x.split('-')[2] for x in test.split()]


# TODO: Freq of dep extractions


# test dep
doc=nlp(u'Countries such as Spain, France and Germany.')
spacy.displacy.render(doc,style='dep')

edges=[]    
for token in doc:
    for child in token.children:
                edges.append(('{0}'.format(token.lower_),'{0}'.format(child.lower_)))

g=nx.Graph(edges)
path=nx.shortest_path(g,source='spain',target='countries')
paths=[]
for word in path:
    for token in doc:
        if word==token.text.lower():
            paths.append(token.text+'-'+token.pos_+'-'+token.dep_)
    
