#!/bin/bash

cd $(dirname $0)/..

target=${1:-all}

function error() {
  printf "\e[31m[ERROR] $1\n"
  exit 1
}

function renderAll() {
  echo "Rendering..."
  mkdir -p output
  pandoc metadata.yaml chapters/*.md \
     -f markdown+footnotes+implicit_figures \
     -V mainfont:"Times New Roman" \
     --pdf-engine=xelatex \
     --columns=100 \
     --toc -s -i -o output/all.pdf
    echo "Finished rendering."
}

case $target in
  all) renderAll;;
  *) renderAll;;
esac
# outputPDF
