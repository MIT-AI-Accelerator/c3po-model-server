#changed lines 15, 21
import gc
import glob
import hashlib
import itertools
import json
import os
import re
import subprocess
import time
from os.path import join as pjoin

import torch
from multiprocess import Pool
from transformers import XLMRobertaTokenizer

from others.logging import logger
from others.utils import clean
from prepro.utils import _get_word_ngrams

import numpy as np

def load_json_english(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            continue
        if (flag):
            tgt.append(tokens)
            flag = False
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt

#changed lines 45-63
def load_json(p, lower): #deal with Arabic Stanford parser output
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (flag):
            tgt.append(tokens)
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@' and tokens[1] == 'highlight'):
            tgt.append(tokens[2:])
            flag = True
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations([i for i in range(len(sents)) if i not in impossible_sents], s + 1)
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if (s == 0 and rouge_score == 0):
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()

#changed line 166
class XLMRobertaRegressionData():
    #changed lines 170-173
    def __init__(self, args):
        self.args = args
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.sep_vid = self.tokenizer._convert_token_to_id(self.tokenizer.sep_token)
        self.cls_vid = self.tokenizer._convert_token_to_id(self.tokenizer.cls_token)
        self.pad_vid = self.tokenizer._convert_token_to_id(self.tokenizer.pad_token)

    #changed lines 176-252
    def preprocess(self, src, tgt):

        if (len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

#        labels = np.array(tgt)[:,1]
        if len(src)>1:
          labels = np.array(tgt)[:,1]
        else:
          labels = np.array(tgt)

        print('%d, %d'%(labels.shape[0], len(src)))
#        assert labels.shape[0] == len(src)
        try:
          assert labels.shape[0] == len(src)
        except:
          print('length mismatch: %d, %d'%(labels.shape[0], len(src)))
          print(src)
          return None 

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]

        src = [src[i][:self.args.max_src_ntokens] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[:self.args.max_nsents]
        labels = labels[:self.args.max_nsents]

        if (len(src) < self.args.min_nsents):
            return None
        if (len(labels) == 0):
            return None

        src_txt = [' '.join(sent) for sent in src]
        # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
        # text = [_clean(t) for t in text]
##        text = ' [SEP] [CLS] '.join(src_txt)
        text = ' </s> <s> '.join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
#        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
        src_subtokens = ['<s>'] + src_subtokens + ['</s>']


        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]

        src_txt = [original_src_txt[i] for i in idxs]
        tgt_txt = src_txt
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt
#        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt, idxs

class XLMRobertaData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        #FIRST set sep, cls, pad token ids; these are set to angle brackets, lower case by default
#        self.sep_vid = self.tokenizer.vocab['[SEP]']
#        self.cls_vid = self.tokenizer.vocab['[CLS]']
#        self.pad_vid = self.tokenizer.vocab['[PAD]']
        self.sep_vid = self.tokenizer._convert_token_to_id(self.tokenizer.sep_token)
        self.cls_vid = self.tokenizer._convert_token_to_id(self.tokenizer.cls_token)
        self.pad_vid = self.tokenizer._convert_token_to_id(self.tokenizer.pad_token)
##        self.sep_vid = self.tokenizer.sep_token_id
##        self.cls_vid = self.tokenizer.cls_token_id
##        self.pad_vid = self.tokenizer.pad_token_id

    #changed lines 281, 285, 301
    def preprocess(self, src, tgt, oracle_ids):

        if (len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        labels = [0] * len(src)
        for l in oracle_ids:
            labels[l] = 1

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]

        src = [src[i][:self.args.max_src_ntokens] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[:self.args.max_nsents]
        labels = labels[:self.args.max_nsents]

        if (len(src) < self.args.min_nsents):
            return None
        if (len(labels) == 0):
            return None

        src_txt = [' '.join(sent) for sent in src]
        # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
        # text = [_clean(t) for t in text]
        text = ' </s> <s> '.join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
#        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
        src_subtokens = ['<s>'] + src_subtokens + ['</s>']


        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt, idxs

def format_to_bert(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert, a_lst):
            pass
        pool.close()
        pool.join()

#changed lines 323-339
def format_to_bert_regression(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert_regression, a_lst):
            pass

        pool.close()
        pool.join()

#changed lines 342-358
def format_to_bert_aiia(args):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((json_f, args, pjoin(args.save_path, real_name.replace('json', 'bert.pt'))))
        print(a_lst)
        pool = Pool(args.n_cpus)
        for d in pool.imap(_format_to_bert_aiia, a_lst):
            pass

        pool.close()
        pool.join()

#changed lines 362, 370-371, 376-380, 382-383, 385
def tokenize(args):
    arabic = args.arabic #boolean; true when language is Arabic, false when English
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    mapping_fn = "mapping_for_corenlp_%s.txt"%hashhex(stories_dir)
    with open(mapping_fn, "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    if arabic:
      cmd_str = 'java edu.stanford.nlp.pipeline.StanfordCoreNLP   -props StanfordCoreNLP-arabic-noparse.properties -annotators tokenize,ssplit -ssplit.newlineIsSentenceBreak always -filelist %s -outputFormat json -outputDirectory %s'%(mapping_fn, tokenized_stories_dir)
    else:
      cmd_str = 'java edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit -ssplit.newlineIsSentenceBreak always -filelist %s -outputFormat json -outputDirectory %s'%(mapping_fn, tokenized_stories_dir)
    command = cmd_str.split()
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    process = subprocess.Popen(command)
    process.wait()
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove(mapping_fn)

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
            tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

#changed lines 403, 406, 409
def _format_to_bert(params):
    json_file, args, save_file = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = XLMRobertaData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file, 'r', encoding='utf-8'))
    datasets = []
    for d in jobs:
        source, tgt, fn = d['src'], d['tgt'], d['f']
        if (args.oracle_mode == 'greedy'):
            oracle_ids = greedy_selection(source, tgt, 3)
        elif (args.oracle_mode == 'combination'):
            oracle_ids = combination_selection(source, tgt, 3)
        b_data = bert.preprocess(source, tgt, oracle_ids)
        if (b_data is None):
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()

#changed lines 427-463
def _format_to_bert_aiia(params):
    json_file, args, save_file = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = XLMRobertaData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))

    datasets = []
    for i,d in enumerate(jobs):
        print('enumerating')
#        source, tgt = d['src'], d['tgt']
        source, tgt, fn = d['src'], d['tgt'], d['f']

        if (args.oracle_mode == 'greedy'):
            oracle_ids = greedy_selection(source, tgt, 3)
        elif (args.oracle_mode == 'combination'):
            oracle_ids = combination_selection(source, tgt, 3)
        b_data = bert.preprocess(source, tgt, oracle_ids)
        if (b_data is None):
            #write out which file was ignored
            logger.info('Ignoring file %d' % i)
            continue
##        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
##        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
##                       'src_txt': src_txt, "tgt_txt": tgt_txt, "fn": fn}
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt, idxs = b_data
        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                'src_txt': src_txt, "tgt_txt": tgt_txt, "fn": fn, "idxs": idxs}
        datasets.append(b_data_dict)
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()

#changed lines 466-499
def _format_to_bert_regression(params):
    json_file, args, save_file = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = XLMRobertaRegressionData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))

    datasets = []
    for i,d in enumerate(jobs):
        print('enumerating')
#        source, tgt = d['src'], d['tgt']
        source, tgt, fn = d['src'], d['tgt'], d['f']

        b_data = bert.preprocess(source, tgt)

        if (b_data is None):
            #write out which file was ignored
            logger.info('Ignoring file %d' % i)
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt, "fn": fn}
#        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt, idxs = b_data
#        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
#                'src_txt': src_txt, "tgt_txt": tgt_txt, "fn": fn, "idxs": idxs}
        datasets.append(b_data_dict)
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()

#changed lines 502-553
def format_to_lines_regression(args):
    corpus_mapping = {}
    corpus_type = 'test'
    test_files = []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        test_files.append(f)

    corpora = {'test': test_files}
    for corpus_type in ['test',]:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        filenames = []
        datasets_with_filenames = []
        p_ct = 0
        for d_with_f in pool.imap_unordered(_format_to_lines_regression, a_lst):
#            d = {k:d_with_f[k] for k in ('src','tgt')}
            d = {'src':d_with_f['src'], 'tgt':d_with_f['tgt'].tolist()}
            f = d_with_f['f']
            dwf = {'src':d_with_f['src'], 'tgt':d_with_f['tgt'].tolist(), 'f':d_with_f['f']}
            dataset.append(d)
            filenames.append(f)
#            datasets_with_filenames.append(d_with_f) #new
            datasets_with_filenames.append(dwf) #new
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w', encoding='utf-8') as save:
##                    save.write(json.dumps(datasets_with_filenames, ensure_ascii=False))
                    save.write(json.dumps(datasets_with_filenames, ensure_ascii=False, separators=(',', ':')))
                    p_ct += 1
                    dataset = []
                    datasets_with_filenames = []
                with open(args.file_order_path, 'a') as order_save:
                    for fn in filenames:
                        order_save.write(fn + '\n')
                    filenames = []
                    datasets_with_filenames = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w', encoding='utf-8') as save:
                save.write(json.dumps(datasets_with_filenames, ensure_ascii=False))
                p_ct += 1
                dataset = []
                datasets_with_filenames = []
            with open(args.file_order_path, 'a') as order_save:
                for fn in filenames:
                    order_save.write(fn + '\n')
                filenames = []
                datasets_with_filenames = []

#changed lines 556-614
def format_to_lines_aiia(args):
    corpus_mapping = {}
    corpus_type = 'test'
    test_files = []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        test_files.append(f)

    corpora = {'test': test_files}
    for corpus_type in ['test',]:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        filenames = []
        datasets_with_filenames = []
        p_ct = 0
#        for d in pool.imap_unordered(_format_to_lines, a_lst):
        for d_with_f in pool.imap_unordered(_format_to_lines_aiia, a_lst):
##        for d_with_f in pool.imap(_format_to_lines_aiia, a_lst):
            d = {k:d_with_f[k] for k in ('src','tgt')}
            f = d_with_f['f']
            dataset.append(d)
            filenames.append(f)
            datasets_with_filenames.append(d_with_f) #new
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
#                with open(pt_file, 'w') as save:
                with open(pt_file, 'w', encoding='utf-8') as save:
                    # save.write('\n'.join(dataset))
#                    save.write(json.dumps(dataset))
##                    save.write(json.dumps(dataset, ensure_ascii=False))
                    save.write(json.dumps(datasets_with_filenames, ensure_ascii=False))
                    p_ct += 1
                    dataset = []
                    datasets_with_filenames = []
                with open(args.file_order_path, 'a') as order_save:
                    for fn in filenames:
                        order_save.write(fn + '\n')
                    filenames = []
                    datasets_with_filenames = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
#            with open(pt_file, 'w') as save:
            with open(pt_file, 'w', encoding='utf-8') as save:
                # save.write('\n'.join(dataset))
#                save.write(json.dumps(dataset))
##                save.write(json.dumps(dataset, ensure_ascii=False))
                save.write(json.dumps(datasets_with_filenames, ensure_ascii=False))
                p_ct += 1
                dataset = []
                datasets_with_filenames = []
#        if (len(filenames) > 0):
            with open(args.file_order_path, 'a') as order_save:
                for fn in filenames:
                    order_save.write(fn + '\n')
                filenames = []
                datasets_with_filenames = []

def format_to_lines(args):
    corpus_mapping = {}
    for corpus_type in ['valid', 'test', 'train']:
        temp = []
        for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        real_name = f.split('/')[-1].split('.')[0]
        if (real_name in corpus_mapping['valid']):
            valid_files.append(f)
        elif (real_name in corpus_mapping['test']):
            test_files.append(f)
        elif (real_name in corpus_mapping['train']):
            train_files.append(f)

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []

#changed lines 660-679
def _format_to_lines_regression(params):
    f, args = params
    prob_f = os.path.join(os.path.join(os.path.dirname(os.path.dirname(f)), 'probabilities'), os.path.basename(f).split('.')[0] + '.txt')
    print(f)
    print(prob_f)
    if args.arabic:
      source, _ = load_json(f, args.lower)
    else:
      source, _ = load_json_english(f, args.lower)
    tgt = np.loadtxt(prob_f) 
    return {'f':f, 'src': source, 'tgt': tgt}

def _format_to_lines_aiia(params):
    f, args = params
#    print(f)
    if args.arabic:
      source, tgt = load_json(f, args.lower)
    else:
      source, tgt = load_json_english(f, args.lower)
    return {'f':f, 'src': source, 'tgt': tgt}

def _format_to_lines(params):
    f, args = params
    print(f)
    source, tgt = load_json(f, args.lower)
    return {'src': source, 'tgt': tgt}
