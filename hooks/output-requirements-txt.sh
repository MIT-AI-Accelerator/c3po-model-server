#!/bin/bash

poetry export -o requirements.txt --without-hashes --with test

# not proud of this, but not sure how else to exclude aiohttp (unused and causing a p1 pipeline failure
sed -i '' '/aiohttp/d' requirements.txt
