#!/bin/sh

pandoc index.md \
  -o out/deep-compression.pptx

pandoc index.md \
  -t beamer \
  -o out/deep-compression.pdf
