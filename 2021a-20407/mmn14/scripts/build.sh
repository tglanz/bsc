#!/bin/bash

script_dir="$realpath $(dirname $0)"
src_dir="$realpath $script_dir/../src"
target_dir="$realpath $script_dir/../target"
main_class="bloomfilter.Program"

cmd=$1
shift;

case $cmd in
    compile) javac $(find $src_dir -name "*.java") -d $target_dir ;;
    run) java -cp $target_dir $main_class "$@" ;;
esac