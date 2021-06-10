#!/bin/bash

script_dir=$(realpath $(dirname $0))
for file in $(find $script_dir -name "*.dot"); do dot -Tpng -O $file; done