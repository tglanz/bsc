#!/bin/sh

docker run -v $(pwd)/search:/search --rm -it --name python python:3.6.15-slim /bin/bash
