#!/bin/bash

argument=$1

function outputLatex() {
  pandoc main.md \
     -f markdown+footnotes \
     -V dir:"rtl" -V theme:"boxes" -V mainfont:"Times New Roman" -V lang:"he-IL" \
     -V navigation:"horizontal" \
     --pdf-engine=xelatex \
     -s -i --slide-level=2 -t beamer -o out.tex
}

function outputPDF() {
  pandoc main.md \
     -f markdown+footnotes+implicit_figures \
     -V theme:"boxes" -V mainfont:"Times New Roman" -V lang:"he-IL" \
     -V navigation:"horizontal" \
     -H beamer-additional-headers.tex \
     --lua-filter tikz.lua \
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
