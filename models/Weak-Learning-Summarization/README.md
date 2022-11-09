# Summarization

This repository contains scripts for a weak learning summarization pipeline in Arabic and English. See Conda Environment Setup below.


Overview:
1. Generate stories (snorkel_compute_labels_languageagnostic.py)
2. Run snorkel pipeline to get labels (snorkel_compute_labels_languageagnostic.py). There is a hard-coded boolean (arabic=True) to switch between Arabic and English.
3. Train XLMRSum on training data (see BertSum-XLMRoberta/README.md for instructions)
4. Download Arabic data from https://sourceforge.net/projects/easc-corpus/ and unzip. Generate test stories (generate_stories_stanfordtokenizer.py) for Arabic data only, since CNNDM stories are available for download.
5. Test XLMRobertaSum on test data (see BertSum-XLMRoberta/README.md for instructions)
6. Assess output (test_bertsum_output.py)
7. Run weak learner assessment (test_summarizers.py)

## Note

Summarization utilizes source code derived from summa library distributed under the MIT License (https://opensource.org/licenses/MIT).  The summa source code can be found at https://github.com/summanlp/textrank.

## Code Set-Up

Prior to running code, copy BertSum-XLMRoberta/src/prepro/data_builder.py file to top level of repository. Copy BertSum-XLMRoberta/src/others/utils.py to top level of repository as bertsum_utils.py

The [text_summarizer](https://github.com/cordelia-io/centroid-summarizer/tree/d0b5bb150d3d94cde2086932787e44365e0b1e52/text_summarizer) package from the [centroid-summarizer](https://github.com/cordelia-io/centroid-summarizer/tree/d0b5bb150d3d94cde2086932787e44365e0b1e52) library is utilized in this code. Add in code from [centroid_bow.py](https://github.com/cordelia-io/centroid-summarizer/blob/d0b5bb150d3d94cde2086932787e44365e0b1e52/text_summarizer/centroid_bow.py) and [base.py]( https://github.com/cordelia-io/centroid-summarizer/blob/d0b5bb150d3d94cde2086932787e44365e0b1e52/text_summarizer/base.py) as described below. Code from external library centroid-summarizer distributed under the [GPL-v3 License](https://github.com/cordelia-io/centroid-summarizer/blob/d0b5bb150d3d94cde2086932787e44365e0b1e52/LICENSE).

Prior to running the code, make the following modifications in weak_summarizers.py. Modify weak_summarizers.py in 4 places:
1. line 41: copy in lines 17-21 from base.py
2. line 175: delete placeholder function definition
3. line 176:
	copy in lines 12, 18, 19 from centroid_bow.py
	
	copy in lines 21-23 from centroid_bow.py
	
	copy in lines 25 from centroid_bow.py; replace second argument with following two arguments: sentences, sentence_embeddings
	
	copy in line 36 from centroid_bow.py; replace name of variable to sum with sentence_embeddings
	
	copy in lines 37-40 from centroid_bow.py
	
	copy in lines 42-45 from centroid_bow.py; replace original name of embedding variable with sentence_embeddings; variable being indexed in second argument of final line should be sentences
	
	copy in line 47 from centroid_bow.py 
	
	copy in lines 49-66 from centroid_bow.py 
	
	copy in line 68 from centroid_bow.py; set the variable equal only to the list
	
4. line 178: copy in line 69 from centroid_bow.py; add ', summary_indices' to end of line

## Conda Environment Setup

1. conda env create -f environment.yml --name summarization
2. Alternative is to use requirements.txt
3. To install pyrouge on Linux, follow the 6-step instructions in https://stackoverflow.com/questions/45894212/installing-pyrouge-gets-error-in-ubuntu (Note that you'll need to have the root right to execute Step 4)

## License
This work is licensed under a
[BSD-2-Clause License][bsd-2-clause].

[bsd-2-clause]: https://opensource.org/licenses/bsd-license.php

## Disclaimer
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

Research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

© 2022 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

SPDX-License-Identifier: BSD-2-Clause
