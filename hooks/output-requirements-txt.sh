#!/bin/bash

poetry export -o requirements.txt --without-hashes --with test
