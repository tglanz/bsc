#!/bin/sh

cd $(dirname $0)

docker run -v $(pwd)/multiagent:/multiagent --rm -it --name python python:3.6.15-slim /bin/bash
