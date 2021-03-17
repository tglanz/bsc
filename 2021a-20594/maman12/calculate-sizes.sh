#!/bin/bash

size $(find /usr/bin /bin/ -executable) 2>/dev/null | cut -f 1 | tail +2 | sort > ./tmp

sum=$(awk '{acc+=$1} END {printf "%d", acc}' ./tmp)
count=$(wc -l ./tmp | cut -f1 -d ' ')
average=$(($sum / $count))
median=$(tail +$((count / 2)) ./tmp | head -n 1)

echo "sum=$sum; count=$count; average=$average; median=$median"
