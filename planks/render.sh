#!/bin/bash
#


# pandoc main.md \
#  -f markdown+footnotes \
#  -V dir:"rtl" -V theme:"boxes" -V mainfont:"Times New Roman" -V lang:"he-IL" \
#  -V navigation:"horizontal" \
#  --pdf-engine=xelatex \
#  -s -i --slide-level=2 -t beamer -o out.pdf

pandoc main.md \
  -V dir:"rtl" -V mainfont:"Times New Roman" -V lang:"he-IL" \
  --pdf-engine=xelatex \
  -s -i --slide-level=2 -t revealjs -o out.html
