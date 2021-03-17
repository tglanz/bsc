pandoc \
    --pdf-engine=xelatex \
    -V mainfont:"Times New Roman" \
    -V dir:rtl \
    ex12.md -o ex12.pdf
