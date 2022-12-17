#!/bin/bash

argument=$1

function outputPDF() {
  pandoc main.md \
     -f markdown+footnotes \
     -V dir:"rtl" -V theme:"boxes" -V mainfont:"Times New Roman" -V lang:"he-IL" \
     -V navigation:"horizontal" \
     --pdf-engine=xelatex \
     -s -i --slide-level=2 -t beamer -o out.pdf
}

function outputHTML() {
  pandoc main.md \
    -V dir:"rtl" -V mainfont:"Times New Roman" -V lang:"he-IL" \
    --pdf-engine=xelatex \
    -s -i --slide-level=2 -t revealjs -o out.html
}

case $argument in
  all)
    outputPDF
    outputHTML
    ;;
  pdf)
    outputPDF
    ;;
  html)
    outputHTML
    ;;
  *) ;;
esac
