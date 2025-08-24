#!/bin/bash
if [ -n "$1" ]; then
    cmake --build build --target cuda_demo -j"$1"
else
    cmake --build build --target cuda_demo
fi