'''
    __version__="1.0"
    __description__ = "Script to run snorkel pipeline for summarization of Arabic and English data"
    __copyright__= "© 2022 MASSACHUSETTS INSTITUTE OF TECHNOLOGY"

    __disclaimer__="THE SOFTWARE/FIRMWARE IS PROVIDED TO YOU ON AN “AS-IS” BASIS."

    __SPDX_License_Identifier__="BSD-2-Clause"
'''

#!/usr/bin/env python

#
# Imports
#
from weak_summarizers import CentroidSentenceBertSummarizer, TextRankSentenceBertSummarizer, SentenceIndexSummarizer, BinarySentenceLengthSummarizer, NumbersSummarizer, BinaryNumbersSummarizer  #, KalimatDefaultSummarizer

from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace 

import pandas
import tqdm

import torch
from sentence_transformers import SentenceTransformer

from snorkel.labeling import labeling_function,LFApplier
from snorkel.labeling.model import LabelModel

import os
import json

from itertools import groupby

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from time import time
import numpy as np

import _pickle as pkl

import tensorflow_datasets as tfds

#
# Labeling Functions
#
@labeling_function()
def centroid_lf(x):
    """ Centroid-based weak summarizer """
    return x.centroid

@labeling_function()
def centroid_repeat_lf(x):
    """ Centroid-based weak summarizer """
    return x.centroid_repeat

@labeling_function()
def centroid_tfidf_lf(x):
    """ Centroid-based, TFIDF embedding weak summarizer """
    return x.centroid_tfidf

@labeling_function()
def centroid_dim100_lf(x):
    """ Centroid-based, dim-100 SentenceBert embedding weak summarizer """
    return x.centroid_dim100

@labeling_function()
def textrank_lf(x):
    """ Text-Rank-based weak summarizer """
    return x.textrank

@labeling_function()
def textrank_tfidf_lf(x):
    """ Text-Rank-based TFIDF embedding weak summarizer """
    return x.textrank_tfidf

@labeling_function()
def textrank_dim100_lf(x):
    """ Text-Rank-based dim-100 SentenceBert embedding weak summarizer """
    return x.textrank_dim100

@labeling_function()
def sentindex_lf(x):
    """ Sentence-position based weak summarizer """
    return x.sentindex

@labeling_function()
def binsentlen_lf(x):
    """ Sentence-position based weak summarizer """
    return x.binsentlen

@labeling_function()
def nums_lf(x):
    """ Sentence-position based weak summarizer """
    return x.nums

@labeling_function()
def binnums_lf(x):
    """ Sentence-position based weak summarizer """
    return x.binnums

@labeling_function()
def kalimat_default_lf(x):
    """ Kalimat default weak summarizer """
    return x.kalimat_default


#
# Helpers
#
def get_device():
    """ 
    Setup GPU and return device 
    Here we are assuming visible GPUs are available 
    (done either via qrsh or CUDA_VISIBLE_DEVICES)
    """

    # GPU should be available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    return device

def load_data(split_arg='val'):
  test_ds_loaded = tfds.load('cnn_dailymail', split=split_arg, shuffle_files=False)

  docs = []
  highlights = [] #list of lists; each sublist is the highlight sentences for a given document

  for doc_index, elt in enumerate(test_ds_loaded):
    if doc_index %10 == 0:
      print(doc_index)
    try:
      doc = [sentence.text for sentence in nlp(elt['article'].decode('utf8')).sentences[:1000]]
    except:
      doc = [sent for sent in elt['article'].decode('utf8').split('.')[:1000] if len(sent.strip())>0] #get rid of sentences of only whitespace
    docs.append(doc)
    highlights.append(elt['highlights'].decode('utf8').split('\n'))

  return (docs, highlights)

if __name__ == "__main__":

    #
    # Setup, Inputs/Outputs
    #
    kalimat_fn = Path('~/FinalDataUTF.csv')
    outdir = Path(os.getcwd())

    fnameout = outdir / "snorkel_sentence_summ_probs.jl"
    fnameout_labels = outdir / "snorkel_sentence_summ_labels.jl"

    #
    # Load, embed data
    #

    #get cuda device (GPU or CPU)
    device = get_device()

    #sentencebert model to embed each sentence
    model = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1', device=device) 

    #
    # Read and Embed Kalimat articles
    #
    print('loading data...')
    arabic = False
    if arabic:
      emp_df = pandas.read_csv(kalimat_fn, usecols=['id','text_x', 'text_y'])
#      docs = [row['text_y'].split() for ctr, row in empf_df.iterrows() if ctr != 1438] #is this too big? 
      docs = [row['text_y'].split('.') for ctr, row in emp_df.iterrows() if ctr != 1438] #is this too big? 
      default_highlights = [row['text_x'].split('.') for ctr, row in emp_df.iterrows() if ctr != 1438] #highlights directly from kalimat automatically generated labels

    else:
      docs, highlights = load_data('train')

    print('finished loading!')

    #
    # Setup Weak Summarizers and Snorkel Labeling Functions
    #
    '''
    weak_summarizers = {'centroid':  CentroidSentenceBertSummarizer(),
                        'textrank':  TextRankSentenceBertSummarizer(),
                        'sentindex': SentenceIndexSummarizer([0]), #first-sentence
                       }
    '''

##    weak_summarizers = {'centroid':  CentroidSentenceBertSummarizer(sim_threshold=0.925),
##                        'textrank':  TextRankSentenceBertSummarizer(0.2),
##                        'sentindex': SentenceIndexSummarizer(range(10)), #first-sentence
##                       }

    weak_summarizers = {'centroid':  CentroidSentenceBertSummarizer(sim_threshold=0.85),
##                        'textrank':  TextRankSentenceBertSummarizer(0.333140),
                        'textrank':  TextRankSentenceBertSummarizer(0.1),
                        'sentindex': SentenceIndexSummarizer(range(3)), #first-sentence
#                        'binsentlen': BinarySentenceLengthSummarizer(), #first-sentence
#                        'nums': NumbersSummarizer(4), #first-sentence
#                        'binnums': BinaryNumbersSummarizer(), #first-sentence
                       }

    '''
    weak_summarizers = {'centroid':  CentroidSentenceBertSummarizer(),
                        'textrank':  TextRankSentenceBertSummarizer(),
                        'sentindex': SentenceIndexSummarizer([0]), #first-sentence
                        'binsentlen': BinarySentenceLengthSummarizer(), #first-sentence
                        'nums': NumbersSummarizer(), #first-sentence
                        'binnums': BinaryNumbersSummarizer(), #first-sentence
                       }
    '''

    weak_labels = {k:defaultdict(str) for k in weak_summarizers}

    #
    # Run weak summarizers over corpus, record output on a per-sentence basis
    #
    corpus = []

    #precompute inverse doc frequency for TFIDF
    tfidf_vectorizer = TfidfVectorizer()
    doc_generator = (sent for doc in docs for sent in doc)
    tfidf_vectorizer.fit(doc_generator)

    start = time()
    for doc_id, doc in enumerate(docs):
        print(doc_id)
        sentence_embeddings = model.encode(doc, show_progress_bar=False) #don't recompute after storing these to run PCA
        tfidf_embeddings = tfidf_vectorizer.transform(doc).toarray() 

        if len(doc) > 1:
            weak_labels['centroid']  = weak_summarizers['centroid'].get_labels(doc, sentence_embeddings)
            weak_labels['textrank']  = weak_summarizers['textrank'].get_labels(sentence_embeddings)
            weak_labels['sentindex'] = weak_summarizers['sentindex'].get_labels(doc)
#            weak_labels['binsentlen'] = weak_summarizers['binsentlen'].get_labels(doc)
#            weak_labels['nums'] = weak_summarizers['nums'].get_labels(doc)
#            weak_labels['binnums'] = weak_summarizers['binnums'].get_labels(doc)
        else:
            weak_labels['centroid'] = {0:1}
            weak_labels['textrank'] = {0:1}
            weak_labels['sentindex'] ={0:1}
#            weak_labels['binsentlen'] ={0:1}
#            weak_labels['nums'] ={0:1}
#            weak_labels['binnums'] ={0:1}

        # Prepare raw labels for snorkel snorkel LabelFunction input format
        sents = [SimpleNamespace(uid=f"{doc_id}_{i}") for i in range(len(doc))] 
        for i in range(len(sents)):
            setattr(sents[i],'doc_i',doc_id)
            setattr(sents[i],'text',doc[i])
            for k in weak_labels:
                setattr(sents[i],k,weak_labels[k][i])

        corpus.extend(sents)
    end = time()
    print('time to embed documents:')
    print(end-start)
    #
    # Snorkel Label Prediction
    #

    #collect and apply labeling functions to 
    lfs = [centroid_lf,textrank_lf,sentindex_lf] #3 classifiers from aia presentation 
#    lfs = [centroid_lf,textrank_lf,sentindex_lf, binsentlen_lf, nums_lf, binnums_lf]
#    lfs = [centroid_lf,textrank_lf,sentindex_lf, nums_lf] #subset for above-chance english classifiers
#    lfs = [centroid_lf,sentindex_lf, binsentlen_lf, nums_lf, binnums_lf] #subset for above-chance arabic classifiers
    applier = LFApplier(lfs)
    L_train = applier.apply(corpus)

    # fit snorkel label model to data
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

    estimated_label_probs = label_model.predict_proba(L_train)
    estimated_labels = label_model.predict(L_train)

    # if you want to see snorkel predicted labels next to weak labelers
    if False:
        for i,x in enumerate(corpus):
            x.snorkel = estimated_label_probs[i,1]

    # write-out predicted snorkel label probabilities
    snorkel_summ_probs = {}
    for i,x in enumerate(corpus):
        snorkel_summ_probs[x.uid] = estimated_label_probs[i,1]

    fo = open(fnameout,"w")
    for i,x in enumerate(corpus):
        fo.write(json.dumps({x.uid:estimated_label_probs[i,1]})+"\n")
    fo.close()

    # write-out predicted snorkel labels
    snorkel_summ_labels = {}
    for i,x in enumerate(corpus):
        snorkel_summ_labels[x.uid] = estimated_labels[i]

    fo = open(fnameout_labels,"w")
    for i,x in enumerate(corpus):
#        fo.write(json.dumps({x.uid:estimated_labels[i]})+"\n")
        fo.write(json.dumps({x.uid:int(estimated_labels[i])})+"\n")
    fo.close()

    #todo: output files in .story format
    story_dir = 'cnndm_estimated_stories_3classifiers_cnndmtunedparams/'
#    story_dir = 'kalimat_estimated_stories_allclassifiers_defaultparams/'

    if not os.path.exists(story_dir):
        os.makedirs(story_dir)

    #group sentences by doc, write to story file
    for doc_i, sents in groupby(enumerate(corpus), lambda tup: tup[1].doc_i):
        doc, labels = zip(*[(sent[1].text,estimated_labels[sent[0]]) for sent in sents])
        full_doc = '.'.join(doc)
        summary = '.\n\n@highlight\n\n'.join([doc[i] for i in range(len(doc)) if labels[i]==1])
        with open(os.path.join(story_dir, 'doc_%d.story'%doc_i), 'w', encoding='utf8') as f:
            f.write(full_doc + '\n\n@highlight\n\n' + summary)

    #TO-DO: (a draft by CKD, please feel to add/detract)
    # [ ] what percentage of weak labels should be abstentions? (maybe a hyper-parameter for unsupervised weak learners)
    # [ ] arabic sentence segmenter? same sentence segmenter needs to be run when fine-tuning over snorkel labels
    # [ ] post-filtering snorkel outputs (i.e. what if no sentence is labeled as a summary, no snorkel score?)

