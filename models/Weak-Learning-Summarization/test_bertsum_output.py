'''
    __version__="1.0"
    __description__ = "Script to evaluate BertSum output with ROUGE-1 score"
    __copyright__= "© 2022 MASSACHUSETTS INSTITUTE OF TECHNOLOGY"

    __disclaimer__="THE SOFTWARE/FIRMWARE IS PROVIDED TO YOU ON AN “AS-IS” BASIS."

    __SPDX_License_Identifier__="BSD-2-Clause"
'''

from others.utils import test_rouge #, rouge_results_to_str
import tempfile
import sys

def get_rouge(save_pred, save_gold):
  with tempfile.TemporaryDirectory() as temp_dir:
#    rouges = evaluate_rouge(temp_dir, save_pred.name, save_gold.name)
    rouges = test_rouge(temp_dir, save_pred, save_gold)

  return rouges["rouge_1_f_score"] 

if __name__ == '__main__':
  save_pred = sys.argv[1]
  save_gold = sys.argv[2]
  print(get_rouge(save_pred, save_gold))

