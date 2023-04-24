#!/bin/sh

cd $(dirname $0)/..

for file in chapters/*; do
  echo "running gramma for $file"
  gramma -m check $file 
done
