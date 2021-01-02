#!/bin/bash

pandoc --pdf-engine=xelatex \
    -V mainfont:"Times New Roman" -V dir:ltr \
    $1.md -o $1.pdf
