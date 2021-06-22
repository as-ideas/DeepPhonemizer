#!/usr/bin/env bash

mkdir docs
cp ../README.md docs/index.md
cp ../CONTRIBUTING.md docs/CONTRIBUTING.md
cp ../LICENSE docs/LICENSE.md
cp -R ../figures docs/
python autogen.py
mkdir ../docs
mkdocs build -c -d ../docs/