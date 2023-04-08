#!/bin/bash

argument=${1:-pdf}

function outputPDF() {
  pandoc metadata.yaml main.md \
     -f markdown+footnotes+implicit_figures \
     -V mainfont:"Times New Roman" \
     --pdf-engine=xelatex \
     --toc -s -i -o out.pdf
}

case $argument in
  all)
    outputPDF
    outputHTML
    outputLatex
    ;;
  pdf)
    outputPDF
    ;;
  html)
    outputHTML
    ;;
  latex)
    outputLatex
    ;;
  *) ;;
esac
