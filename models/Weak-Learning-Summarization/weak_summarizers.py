'''
    __version__="1.0"
    __description__ = "Script defining summarizer classes for weak learners"
    __copyright__= "© 2022 MASSACHUSETTS INSTITUTE OF TECHNOLOGY"

    __disclaimer__="THE SOFTWARE/FIRMWARE IS PROVIDED TO YOU ON AN “AS-IS” BASIS."

    __SPDX_License_Identifier__="BSD-2-Clause"
'''

#!/usr/bin/env python

#
# Imports
#
import numpy as np
from scipy.spatial.distance import cosine

#text rank
from summa.pagerank_weighted import pagerank_weighted_scipy as pagerank
from summa.commons import build_graph as summa_build_graph
from summa.commons import remove_unreachable_nodes as summa_remove_unreachable_nodes

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist, squareform
import scipy

from statistics import median
import re
import random

# Define the label mappings for convenience
SUMM = 1
NOT_SUMM = 0
ABSTAIN = -1

#
# Helpers
#

#modify code here

#
# Summarizer Classes
#

class RandomSummarizer():
    def __init__(self, summ_len=3):
        self.summ_len = summ_len 
        return

    def summarize(self, sentences):
        summary_inds = random.sample(list(range(len(sentences))), min(len(sentences),self.summ_len))
        return ([sentences[i] for i in summary_inds], summary_inds) 

    def get_labels(self, sentences):
        """ return sentence-level labels for sentences predicted to be summary-sentences """
        _, summ_indices = self.summarize(sentences)

        sent_labels = {}
        for i in range(len(sentences)):
            if i in summ_indices:
                sent_labels[i] = SUMM
            else:
                sent_labels[i] = NOT_SUMM

        return sent_labels

class BinaryNumbersSummarizer():
    def __init__(self):
        return

    def find_numbers(self, input_string): #adapted from https://stackoverflow.com/questions/19859282/check-if-a-string-contains-a-number 
        return bool(re.search(r'\d', input_string))

    def summarize(self, sentences):
        summary_inds = [i for i,sent in enumerate(sentences) if self.find_numbers(sent)]
        return ([sentences[i] for i in summary_inds], summary_inds) 

    def get_labels(self, sentences):
        """ return sentence-level labels for sentences predicted to be summary-sentences """
        summ_indices = self.summarize(sentences)[1]

        sent_labels = {}
        for i in range(len(sentences)):
            if i in summ_indices:
                sent_labels[i] = SUMM
            else:
                sent_labels[i] = NOT_SUMM

        return sent_labels

class NumbersSummarizer():
    def __init__(self, summary_length=3):
        self.summary_length = summary_length
        return

    def summarize(self, sentences):
        numbers_counts = [len(re.findall(r'\d', sent)) for sent in sentences]
        max_count = max(numbers_counts)
        numbers_scores = [count/max_count if max_count>0 else 0 for count in numbers_counts]
        summary_inds = [tup[0] for tup in sorted(enumerate(numbers_scores), key=lambda tup:tup[1])[-self.summary_length:]]
        return ([sentences[i] for i in summary_inds], summary_inds) 

    def get_labels(self, sentences):
        """ return sentence-level labels for sentences predicted to be summary-sentences """
        summ_indices = self.summarize(sentences)[1]

        sent_labels = {}
        for i in range(len(sentences)):
            if i in summ_indices:
                sent_labels[i] = SUMM
            else:
                sent_labels[i] = NOT_SUMM

        return sent_labels

class BinarySentenceLengthSummarizer():
    def __init__(self):
        return

    def summarize(self, sentences):
        lengths = [len(sent.split()) for sent in sentences]
#        lengths_sorted = sorted(enumerate(lengths), key=lambda tup:tup[1])
        lengths_sorted = sorted(lengths)
        half = int(len(sentences)/2.0)
#        q1 = median(lengths_sorted[:len(sentences)])
#        q3 = median(lengths_sorted[len(sentences):])
        q1 = median(lengths_sorted[:half])
        q3 = median(lengths_sorted[-half:])
        summary_inds = [i for i in range(len(sentences)) if lengths[i]>=q1 and lengths[i]<=q3]
        return ([sentences[i] for i in summary_inds], summary_inds)

    def get_labels(self, sentences):
        """ return sentence-level labels for sentences predicted to be summary-sentences """
        summary, summ_indices = self.summarize(sentences)

        sent_labels = {}
        for i in range(len(sentences)):
            if i in summ_indices:
                sent_labels[i] = SUMM
            else:
                sent_labels[i] = NOT_SUMM

        return sent_labels


class KalimatDefaultSummarizer():
    def __init__(self):
        return

    def get_labels(self, sentences, highlights):
        """ return sentence-level labels for sentences predicted to be summary-sentences """
        sentences = [sent.strip() for sent in sentences]
        highlight_set = set([h.strip() for h in highlights])

        #todo: closest pyrouge distance?
        summ_indices = set() 
        for j in range(len(highlights)):
          summ_indices.add(max([(i,evaluate_rouge([[sentences[i]]], [[highlights[j]]], rouge_args=[])) for i in range(len(sentences))], key=lambda tup:tup[1])[0])
#        summ_indices = [i for i, e in enumerate(sentences) if e in highlight_set]

        sent_labels = {}
        for i in range(len(sentences)):
            if i in summ_indices:
                sent_labels[i] = SUMM
            else:
                sent_labels[i] = NOT_SUMM

        return sent_labels


class CentroidSentenceBertSummarizer():

    def placeholder(): #modify code here
        #modify code here
        summary_indices = [s[0] for s in sentences_summary]
        #modify code here
        
    def get_labels(self, sentences, sentence_embeddings, limit_type='word', limit=100):
        """ return sentence-level labels for sentences predicted to be summary-sentences """
        _,summ_indices = self.summarize(sentences,sentence_embeddings,limit_type='word',limit=100)

        sent_labels = {}
        for i in range(len(sentences)):
            if i in summ_indices:
                sent_labels[i] = SUMM
            else:
                sent_labels[i] = NOT_SUMM

        return sent_labels


class TextRankSentenceBertSummarizer():
    ''' based on CentroidSentenceVertSummarizer 
        also on TextRank package's summarize() function
        Perform TextRank over all sentences in document
    '''
    def __init__(self,ratio=0.2):
        self.ratio = ratio

    def _get_similarity(self,sentence_1, sentence_2):
        ''' Get similarity bw sentence representations '''
        v1 = self.sentence_embeddings[int(sentence_1)]
        v2 = self.sentence_embeddings[int(sentence_2)]
        similarity = 0.0
        if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
            similarity = ((1 - cosine(v1, v2)) + 1) / 2
        return similarity

    def _create_valid_graph(self,graph):
        nodes = graph.nodes()
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue
                edge = (nodes[i], nodes[j])
                if graph.has_edge(edge):
                    graph.del_edge(edge)
                graph.add_edge(edge, 1)

    def _set_graph_edge_weights(self,graph):
        ''' Make graph edge weights based on sentence embedding features '''
        for sentence_1 in graph.nodes():
            for sentence_2 in graph.nodes():
                edge = (sentence_1, sentence_2)
                if sentence_1 != sentence_2 and not graph.has_edge(edge):
                    similarity = self._get_similarity(sentence_1, sentence_2)
                    if similarity != 0:
                        graph.add_edge(edge, similarity)

        # Handles the case in which all similarities are zero.
        # The resultant summary will consist of random sentences.
        if all(graph.edge_weight(edge) == 0 for edge in graph.edges()):
            self._create_valid_graph(graph)


    def compute_pagerank(self, sentence_embeddings):
        """ Compute pagerank scores over each sentence embedding in document """
        self.sentence_embeddings = sentence_embeddings
        graph = summa_build_graph([i for i in range(sentence_embeddings.shape[0])])
        self._set_graph_edge_weights(graph)

        # Remove all nodes with all edges weights equal to zero.
        summa_remove_unreachable_nodes(graph)

        # PageRank cannot be run in an empty graph.
        if len(graph.nodes()) == 0:
            return

        # Ranks the tokens using the PageRank algorithm. Returns dict of sentence ind -> pagerank score
        pagerank_scores = pagerank(graph)
        pagerank_scores = {int(k): v for k, v in sorted(pagerank_scores.items(), key=lambda item: item[1],reverse=True)}

        self.pagerank_scores = pagerank_scores


    def summarize(self, sentences, sentence_embeddings):
        """ Derive extractive summaries from sentences """
        self.compute_pagerank(sentence_embeddings)

        #sorted_pr_inds = [int(x) for x in self.pagerank_scores.keys()]
        sorted_pr_inds = list(self.pagerank_scores.keys())

        # Extract the most important sentences with the selected criterion.
#        num_sent_to_extract = len(sentences)*self.ratio
        num_sent_to_extract = int(len(sentences)*self.ratio)
        summary_inds = sorted_pr_inds[:num_sent_to_extract]
#        summary_sentences = sentences[:summary_inds]
        summary_sentences = [sentences[i] for i in summary_inds]

        return summary_sentences,summary_inds


    def get_labels(self,sentence_embeddings):
        """ return sentence-level labels for sentences predicted to be summary-sentences """
        self.compute_pagerank(sentence_embeddings)

        num_to_extract = sentence_embeddings.shape[0]*self.ratio

        sorted_pr = {k:SUMM if i < num_to_extract else NOT_SUMM for i,(k,v) in enumerate(self.pagerank_scores.items())}

        #add back in NOT_SUMM for nodes that were removed
        sorted_pr.update({k:NOT_SUMM for k in range(len(sentence_embeddings)) if k not in sorted_pr})

        return {k: v for k, v in sorted(sorted_pr.items(), key=lambda item: item[0])}


class SentenceIndexSummarizer():
    ''' A basic heuristic summarizer based on target indices (i.e. first sentence, first-three sentences, etc.)
    '''
    def __init__(self,indices):

        if isinstance(indices,int): indices = [indices]

        self.indices = indices


    def summarize(self, sentences):
        """ Derive extractive summaries from sentences """

#        return sentences[indices],indices
#        return sentences[self.indices],self.indices
##        return [sentences[i] for i in self.indices],self.indices
        indices = [i for i in self.indices if i < len(sentences)]
        return [sentences[i] for i in indices],indices


    def get_labels(self,sentences):
        """ return sentence-level labels for sentences predicted to be summary-sentences """

        labels = {}
        for i,x in enumerate(sentences):
            if i == 0:
                labels[i] = SUMM
            else:
                labels[i] = ABSTAIN

        return labels

