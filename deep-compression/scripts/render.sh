#!/bin/bash

cd $(dirname $0)/..

target=${1:-all}

function error() {
  printf "\e[31m[ERROR] $1\n"
  exit 1
}

function renderAll() {
  mkdir -p output
  pandoc metadata.yaml chapters/*.md \
     -f markdown+footnotes+implicit_figures \
     -V mainfont:"Times New Roman" \
     --pdf-engine=xelatex \
     --toc -s -i -o output/all.pdf
}

function renderPruning() {
  mkdir -p output
  pandoc chapters/*pruning.md chapters/*references.md \
     -f markdown+footnotes+implicit_figures \
     -V mainfont:"Times New Roman" \
     --pdf-engine=xelatex \
     --toc -s -i -o output/pruning.pdf
}

case $target in
  all) renderAll;;
  pruning) renderPruning;;
  *) error "unknown target: $target"
esac
# outputPDF
