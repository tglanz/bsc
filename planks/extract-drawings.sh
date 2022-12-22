#!/bin/bash

source_path=tikz.pdf
out_dir=drawings

# drawings_count=2
# 
# for (( i=1; i<=drawings_count; i++ )); do 
#   output_path=drawings/$i.svg
#   pdf2svg $source_path $output_path $i 
# done

convert $source_path $out_dir/image.png
