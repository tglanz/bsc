#!/bin/bash

name=$1

pandoc \
    --pdf-engine=xelatex \
    -V mainfont:"Times New Roman" \
    -V dir:rtl \
    $name.md -o $name.pdf
