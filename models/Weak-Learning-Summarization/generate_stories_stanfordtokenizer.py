'''
    __version__="1.0"
    __description__ = "Script to generate stories from text data in Arabic"
    __copyright__= "© 2022 MASSACHUSETTS INSTITUTE OF TECHNOLOGY"

    __disclaimer__="THE SOFTWARE/FIRMWARE IS PROVIDED TO YOU ON AN “AS-IS” BASIS."

    __SPDX_License_Identifier__="BSD-2-Clause"
'''

import os
import subprocess
import json

#preprocess EASC data
#generate a new story file for each MTurk annotation
articles_dir = 'EASC/EASC-UTF-8/Articles/'
annotations_dir = 'EASC/EASC-UTF-8/MTurk/'
#EASC/EASC-UTF-8/Articles/Topic100
#EASC/EASC-UTF-8/MTurk/Topic59

topic2article = {} #map from 'Topic%d' to article text
for ctr, article_dir in enumerate(os.listdir(articles_dir)):
  contents = os.listdir(os.path.join(articles_dir, article_dir))
  if len(contents) == 0:
    print(article_dir)
  for fn in os.listdir(os.path.join(articles_dir, article_dir)):
    with open(os.path.join(os.path.join(articles_dir, article_dir), fn), 'r', encoding='utf8') as f:
      topic2article[article_dir] = f.read() 

stories_dir = 'easc_stories_stanfordtokenized'
if not os.path.exists(stories_dir):
  os.makedirs(stories_dir)

for annotation_dir in os.listdir(annotations_dir):
#  outdir = os.path.join(stories_dir, annotation_dir)
  outdir = stories_dir
  if not os.path.exists(outdir):
    os.makedirs(outdir)

  for fn in os.listdir(os.path.join(annotations_dir, annotation_dir)):
    with open(os.path.join(os.path.join(annotations_dir, annotation_dir), fn), 'r', encoding='utf8') as f:
      summary = f.read()
    #tokenize summaries:
    if True: 
      highlights_split = []
      with open('temp', 'w', encoding='utf8') as f:
        f.write(summary)
      cmd_str = 'java edu.stanford.nlp.pipeline.StanfordCoreNLP   -props BertSum-XLMRoberta/src/StanfordCoreNLP-arabic-noparse.properties -annotators tokenize,ssplit -ssplit.newlineIsSentenceBreak always -file temp -outputFormat json -outputDirectory temp_dir'
      command = cmd_str.split()
      process = subprocess.Popen(command)
      process.wait()
      d = json.load(open('temp_dir/temp.json', 'r', encoding='utf8'))
      for sent in d['sentences']:
        highlights_split.append(' '.join([t['word'] for t in sent['tokens']]))

    with open(os.path.join(outdir, '_'.join([annotation_dir, fn + '.story'])), 'w', encoding='utf8') as fout: 
      fout.write(topic2article[annotation_dir].strip() + '\n\n@highlight\n\n')
      fout.write('\n\n@highlight\n\n'.join(highlights_split))

