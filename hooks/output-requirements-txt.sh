#!/bin/bash

pipenv requirements > requirements.txt
echo "tensorflow==2.12.0" >> requirements.txt
