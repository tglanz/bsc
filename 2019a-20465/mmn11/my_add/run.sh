#!/bin/bash
name=$1
cat ./$1.input | ./my_add > ./$1.output
