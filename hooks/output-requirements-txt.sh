#!/bin/bash

poetry export -o requirements.txt --without-hashes --with dev
echo "tensorflow==2.11.1" >> requirements.txt
