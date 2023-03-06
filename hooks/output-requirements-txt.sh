#!/bin/bash

pipenv requirements > requirements.txt
echo "tensorflow==2.11.0" >> requirements.txt
