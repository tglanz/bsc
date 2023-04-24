#!/bin/bash

cd $(dirname $0)/..

function outputPDF() {
  pandoc metadata.yaml chapters/*.md \
     -f markdown+footnotes+implicit_figures \
     -V mainfont:"Times New Roman" \
     --pdf-engine=xelatex \
     --toc -s -i -o the-lottery-ticket-hypothesis.pdf
}

outputPDF
