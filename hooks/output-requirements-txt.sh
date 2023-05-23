#!/bin/bash

pipenv requirements > requirements.txt
echo "tensorflow==2.11.1" >> requirements.txt
sed -i 's/^.*transformers.*$/transformers @ git+https://github.com/huggingface/transformers.git@80ca92470938bbcc348e2d9cf4734c7c25cb1c43/' requirements.txt
