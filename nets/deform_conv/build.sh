#!/usr/bin/env bash

PYTHON=${PYTHON:-"python3.6"}

if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
