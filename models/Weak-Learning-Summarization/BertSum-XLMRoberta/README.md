# BertSum

**This code is based on the code for the paper `Fine-tune BERT for Extractive Summarization`**(https://arxiv.org/pdf/1903.10318.pdf). The repository for the original code is here: https://github.com/nlpyang/BertSum.

**Python version**: This code is in Python3.6

**Package Requirements**: pytorch pytorch_pretrained_bert tensorboardX multiprocess pyrouge

Some codes are borrowed from ONMT(https://github.com/OpenNMT/OpenNMT-py)

## Data Preparation 

#### Step 1. Generate Stories
If using CNN/DailyMail corpus: Download and unzip the `stories` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. Put all  `.story` files in one directory (e.g. `../raw_stories`)

Otherwise, generate files (one file for each original article file) in .story format: original article text, followed by '\n\n@highlight\n\n + SUMMARY_SENT' for each summary sentence in the ground-truth summary for that article.

####  Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Download the Arabic model as well and copy it to the same directory. Then add the following command to your bash_profile:
```
export CLASSPATH=/path/to/stanford-corenlp-full-2017-06-09/*
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-full-2017-06-09` directory. 

####  Step 3. Sentence Splitting and Tokenization

If running on Arabic data, add --arabic flag.

```
python preprocess.py -mode tokenize -raw_path RAW_PATH -save_path TOKENIZED_PATH
```

e.g.:
```
python preprocess.py -mode tokenize -raw_path ~/kalimat_estimated_stories_stanfordtokenized/ -save_path ~/kalimat_estimated_stories_stanfordtokenized_bertsumtokenized/ -log_file ../logs/cnndm.log
```

* `RAW_PATH` is the directory containing story files (`../raw_stories`), `JSON_PATH` is the target directory to save the generated json files (`../merged_stories_tokenized`)


####  Step 4. Format to Simpler Json Files
 
If running on Arabic data, add --arabic flag.
 
```
mkdir JSON_PATH
python preprocess.py -mode format_to_lines_aiia -raw_path RAW_PATH -save_path JSON_PATH -map_path MAP_PATH -lower 
mv JSON_PATH* JSON_PATH
```

e.g., 
```
mkdir kalimat_estimated_stories_stanfordtokenized_json
python preprocess.py -mode format_to_lines_aiia -raw_path ~/kalimat_estimated_stories_stanfordtokenized_bertsumtokenized/ -save_path ~/kalimat_estimated_stories_stanfordtokenized_json -map_path ../urls/ -lower -log_file ../logs/cnndm.log
mv ~/kalimat_estimated_stories_stanfordtokenized_json* ~/kalimat_estimated_stories_stanfordtokenized_json
```

* `RAW_PATH` is the directory containing tokenized files (`../merged_stories_tokenized`), `JSON_PATH` is the target directory to save the generated json files (`../json_data/cnndm`), `MAP_PATH` is the  directory containing the urls files (`../urls`)

####  Step 5. Format to PyTorch Files

If running on Arabic data, add --arabic flag.

```
mkdir BERT_DATA_PATH
python preprocess.py -mode format_to_bert_aiia -raw_path JSON_PATH -save_path BERT_DATA_PATH -oracle_mode greedy -n_cpus 4 -log_file ../logs/preprocess.log
```

e.g.,
```
mkdir ~/kalimat_estimated_stories_stanfordtokenized_bertdata
python preprocess.py -mode format_to_bert_aiia -raw_path ~/kalimat_estimated_stories_stanfordtokenized_json/ -save_path ~/kalimat_estimated_stories_stanfordtokenized_bertdata -oracle_mode greedy -n_cpus 4 -log_file ../logs/preprocess.log
```

* `JSON_PATH` is the directory containing json files (`../json_data`), `BERT_DATA_PATH` is the target directory to save the generated binary files (`../bert_data`)

* `-oracle_mode` can be `greedy` or `combination`, where `combination` is more accurate but takes much longer time to process 

#### Step 6. Move .test.* files to .train.* files

Rename files .test.* in `BERT_DATA_PATH` (~/kalimat_estimated_stories_stanfordtokenized_bertdata/) to .train.* 

```
cd BERT_DATA_PATH
for F in *_*; do mv -- "${F}" "${F%test*}train${F##*test}"; done
```

## Model Training

**First run**: For the first time, you should use single-GPU, so the code can download the BERT model. Change ``-visible_gpus 0,1,2  -gpu_ranks 0,1,2 -world_size 3`` to ``-visible_gpus 0  -gpu_ranks 0 -world_size 1``, after downloading, you could kill the process and rerun the code with multi-GPUs.


To train the BERT+Transformer model, run:
```
python train.py -mode train -encoder transformer -dropout 0.1 -bert_data_path ~/kalimat_estimated_stories_stanfordtokenized_bertdata/kalimat_estimated_stories_stanfordtokenized_bert.pt -model_path ../models/kalimat_estimated_bert_transformer -lr 2e-3 -visible_gpus 0  -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 50000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file ../logs/bert_transformer -use_interval true -warmup_steps 10000 -ff_size 2048 -inter_layers 2 -heads 8
```

* `-mode` can be {`train, validate, test`}, where `validate` will inspect the model directory and evaluate the model for each newly saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use

## Model Evaluation

Preprocess EASC test dataset for evaluation:

```
mkdir easc_stories_stanfordtokenized_bertsumtokenized
python preprocess.py -mode tokenize -raw_path ~/easc_data/easc_stories_stanfordsummarysplit/ -save_path ~/easc_data/easc_stories_stanfordtokenized_bertsumtokenized/ -log_file ../logs/cnndm.log

python preprocess.py -mode format_to_lines_aiia -raw_path ~/easc_data/easc_stories_stanfordsummarysplit_bertsumtokenized/ -save_path ~/easc_data/easc_stories_stanfordsummarysplit_json -map_path ../urls/ -lower -log_file ../logs/cnndm.log
mkdir ~/easc_data/easc_stories_stanfordsummarysplit_json
mv ~/easc_data/easc_stories_stanfordsummarysplit_json* ~/easc_data/easc_stories_stanfordsummarysplit_json
mkdir ~/easc_data/easc_stories_stanfordsummarysplit_bertdata
python preprocess.py -mode format_to_bert_aiia -raw_path ~/easc_data/easc_stories_stanfordsummarysplit_json/ -save_path ~/easc_data/easc_stories_stanfordsummarysplit_bertdata -oracle_mode greedy -n_cpus 4 -log_file ../logs/preprocess.log
mv ~/easc_data/easc_stories_stanfordsummarysplit_bertdata/easc_stories_stanfordsummarysplit_bert.pt.test.0.bert.pt  ~/easc_data/easc_stories_stanfordsummarysplit_bertdata/easc_stories_stanfordsummarysplit_bert.test.pt
```

After the training finished, run
```
python train.py -mode validate -bert_data_path ../bert_data/cnndm -model_path MODEL_PATH  -visible_gpus 0  -gpu_ranks 0 -batch_size 30000  -log_file LOG_FILE  -result_path RESULT_PATH -test_all -block_trigram true
```

e.g., 
```
python train.py -mode test -bert_data_path ~/easc_data/easc_stories_stanfordsummarysplit_bertdata/easc_stories_stanfordsummarysplit_bert -test_from ../models/kalimat_estimated_bert_transformer/model_step_50000.pt  -visible_gpus 0  -gpu_ranks 0 -batch_size 30000  -log_file ../logs/test.log  -result_path ~/easc_data/easc_kalimat_estimated_results -test_all -block_trigram true -report_rouge false
```

* `MODEL_PATH` is the directory of saved checkpoints
* `RESULT_PATH` is where you want to put decoded summaries (default `../results/cnndm`)


