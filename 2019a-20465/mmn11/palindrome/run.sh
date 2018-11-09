#!/bin/bash
name=$1

cat ./$1.input | ./palindrome > ./$1.output
