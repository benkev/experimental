#!/bin/bash
file_name=$0
cur_dir="$(pwd)""${file_name:1}"
echo $cur_dir
cat $cur_dir
gcc -std=c99 -o bandwidth bandwidth.c -O3 -lOpenCL -lm
