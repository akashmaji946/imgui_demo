#!/bin/bash
if [ -n "$1" ]; then
    ./build/cuda_demo "$1"
else
    ./build/cuda_demo
fi