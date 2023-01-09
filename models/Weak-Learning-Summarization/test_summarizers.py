'''
    __version__="1.0"
    __description__ = "Script to evaluate individual weak learners using ROUGE-1 score"
    __copyright__= "© 2022 MASSACHUSETTS INSTITUTE OF TECHNOLOGY"

    __disclaimer__="THE SOFTWARE/FIRMWARE IS PROVIDED TO YOU ON AN “AS-IS” BASIS."

    __SPDX_License_Identifier__="BSD-2-Clause"
'''

from weak_summarizers import BinarySentenceLengthSummarizer, BinaryNumbersSummarizer, NumbersSummarizer, SentenceIndexSummarizer, TextRankSentenceBertSummarizer, CentroidSentenceBertSummarizer, RandomSummarizer

from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace 

import pandas
import tqdm

import torch
from sentence_transformers import SentenceTransformer

import os
import json

from itertools import groupby

from data_builder import load_json

from sklearn.feature_extraction.text import TfidfVectorizer

#rouge evaluation imports:
from rouge_utils.utils import test_rouge, rouge_results_to_str
#from rouge_utils import evaluate_rouge #, rouge_results_to_str
#from rouge_utils.eval_logging import init_logger
import tempfile
from others.utils import test_rouge, rouge_results_to_str

def bertsum_rouge(all_docs, all_highlights, all_predicted_indices):
#  save_pred = tempfile.NamedTemporaryFile()
#  save_gold = tempfile.NamedTemporaryFile()
  pred_path = 'pred_path'
  gold_path = 'gold_path'
  save_pred = open(pred_path, 'w', encoding='utf8') 
  save_gold = open(gold_path, 'w', encoding='utf8') 

  for doc, highlights, predicted_indices in zip(all_docs, all_highlights, all_predicted_indices): #for each doc

    #now run pyrouge comparing gold, candidate:
#    gold = '<q>'.join([' '.join(tt) for tt in highlights])
    gold = '<q>'.join(highlights)
    pred = '<q>'.join([doc[i] for i in predicted_indices])
    save_gold.write(gold.strip()+'\n')
    save_pred.write(pred.strip()+'\n')

  #after all docs complete:
  save_pred.close()
  save_gold.close()

  with tempfile.TemporaryDirectory() as temp_dir:
#    rouges = test_rouge(temp_dir, save_pred, save_gold)
    rouges = test_rouge(temp_dir, pred_path, gold_path)

  return rouges["rouge_1_f_score"] 


def pyrouge_f1_score(all_docs, all_highlights, all_predicted_indices):

  ''' Objective function to maximize; returns rouge-1 f-score; has been replaced by rouge_f1_score '''

  save_pred = tempfile.NamedTemporaryFile()
  save_gold = tempfile.NamedTemporaryFile()

  for doc, highlights, predicted_indices in zip(all_docs, all_highlights, all_predicted_indices): #for each doc

    #now run pyrouge comparing gold, candidate:
    pred = [' '.join([doc[i] for i in predicted_indices])]
    gold = [' '.join(highlights)]

    if len(predicted_indices) == 0:
      print('no summary found:')
      print(pred)

    for i in range(len(pred)):
			    save_pred.write(bytes(pred[i].strip()+'\n', encoding='utf8'))
    for i in range(len(gold)):
			    save_gold.write(bytes(gold[i].strip()+'\n', encoding='utf8'))

  save_pred.seek(0)
  save_gold.seek(0)

  with tempfile.TemporaryDirectory() as temp_dir:
#    rouges = evaluate_rouge(temp_dir, save_pred.name, save_gold.name)
    rouges = test_rouge(temp_dir, save_pred.name, save_gold.name)

  save_pred.close()
  save_gold.close()

  return rouges["rouge_1_f_score"] 

if __name__ == '__main__':

    csbs = CentroidSentenceBertSummarizer(0.85) #default: no param
    trsbs = TextRankSentenceBertSummarizer(0.1) #default: no param
    csbs_tfidf = CentroidSentenceBertSummarizer()
    trsbs_tfidf = TextRankSentenceBertSummarizer()
    sis = SentenceIndexSummarizer([0, 1, 2]) #default: [0]
##    ns = NumbersSummarizer(4) #what should param be?
    ns = NumbersSummarizer(4) #default: no param (3) 
    bns = BinaryNumbersSummarizer()
    bsls = BinarySentenceLengthSummarizer() 
    rs = RandomSummarizer() 

    #get cuda device (GPU or CPU)
#    device = get_device()

    #sentencebert model to embed each sentence
#    model = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1', device=device) 
    print('loading model')
    model = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1', device='cpu') 
#    model = SentenceTransformer('xlm-roberta-base', device='cpu') 
    print('done loading')

    #change input to easc corpus
    easc_dir = 'easc_stories_stanfordtokenized'

    tfidf_vectorizer = TfidfVectorizer()
    doc_generator = (' '.join([' '.join(sent) for sent in load_json(os.path.join(easc_dir,fn), lower=True)[0]]) for fn in os.listdir(easc_dir))
    tfidf_vectorizer.fit(doc_generator)

    all_docs = []
    all_highlights = []
    all_csbs_indices = [] #predicted indices, csbs
    all_trsbs_indices = [] #predicted indices, trsbs
    all_csbs_tfidf_indices = [] #predicted indices, csbs
    all_trsbs_tfidf_indices = [] #predicted indices, trsbs
    all_sis_indices = [] #predicted indices, sis
    all_ns_indices = [] #predicted indices, ns
    all_bns_indices = [] #predicted indices, bns
    all_bsls_indices = [] #predicted indices, bsls
    all_rs_indices = [] #predicted indices, rs

    for doc_i, fn in enumerate(os.listdir(easc_dir)):
        if doc_i % 10 ==  0:
          print('up to doc %d'%doc_i)

        f = os.path.join(easc_dir, fn)
        sentences, tgt = load_json(f, lower=True) #does it matter if lower?
        sentences = [' '.join(sent) for sent in sentences]
        highlights = [' '.join(sent) for sent in tgt]

        all_docs.append(sentences)
        all_highlights.append(highlights)

        sentence_embeddings = model.encode(sentences, show_progress_bar=False)
        tfidf_embeddings = tfidf_vectorizer.transform(sentences).toarray() 

        _, csbs_summary_indices = csbs.summarize(sentences, sentence_embeddings)
        _, trsbs_summary_indices = trsbs.summarize(sentences, sentence_embeddings)
        _, csbs_tfidf_summary_indices = csbs_tfidf.summarize(sentences, tfidf_embeddings)
        _, trsbs_tfidf_summary_indices = trsbs_tfidf.summarize(sentences, tfidf_embeddings)
        _, sis_summary_indices = sis.summarize(sentences)
        _, ns_summary_indices = ns.summarize(sentences)
        _, bns_summary_indices = bns.summarize(sentences)
        _, bsls_summary_indices = bsls.summarize(sentences)
        _, rs_summary_indices = rs.summarize(sentences)

        all_csbs_indices.append(csbs_summary_indices)
        all_trsbs_indices.append(trsbs_summary_indices)
        all_csbs_tfidf_indices.append(csbs_tfidf_summary_indices)
        all_trsbs_tfidf_indices.append(trsbs_tfidf_summary_indices)
        all_sis_indices.append(sis_summary_indices)
        all_ns_indices.append(ns_summary_indices)
        all_bns_indices.append(bns_summary_indices)
        all_bsls_indices.append(bsls_summary_indices)
        all_rs_indices.append(rs_summary_indices)

#get pyrouge score on document set:

csbs_f1_score = pyrouge_f1_score(all_docs, all_highlights, all_csbs_indices)
trsbs_f1_score = pyrouge_f1_score(all_docs, all_highlights, all_trsbs_indices)
csbs_tfidf_f1_score = pyrouge_f1_score(all_docs, all_highlights, all_csbs_tfidf_indices)
trsbs_tfidf_f1_score = pyrouge_f1_score(all_docs, all_highlights, all_trsbs_tfidf_indices)
sis_f1_score = pyrouge_f1_score(all_docs, all_highlights, all_sis_indices)
ns_f1_score = pyrouge_f1_score(all_docs, all_highlights, all_ns_indices)
bns_f1_score = pyrouge_f1_score(all_docs, all_highlights, all_bns_indices)
bsls_f1_score = pyrouge_f1_score(all_docs, all_highlights, all_bsls_indices)
rs_f1_score = pyrouge_f1_score(all_docs, all_highlights, all_rs_indices)

print('csbs ROUGE-1: %f'%csbs_f1_score)
print('trsbs ROUGE-1: %f'%trsbs_f1_score)
print('csbs tfidf ROUGE-1: %f'%csbs_tfidf_f1_score)
print('trsbs tfidf ROUGE-1: %f'%trsbs_tfidf_f1_score)
print('sis ROUGE-1: %f'%sis_f1_score)
print('ns ROUGE-1: %f'%ns_f1_score)
print('bns ROUGE-1: %f'%bns_f1_score)
print('bsls ROUGE-1: %f'%bsls_f1_score)
print('rs ROUGE-1: %f'%rs_f1_score)


