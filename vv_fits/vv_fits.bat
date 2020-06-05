#!/bin/bash
#
# vv_fits.bat
#
# Run vv_fits in parallel on several gpu fits files
# Execute with one wildcard parameter generating
# several files, like
#
# $ bash vv_fits.bat *_gpubox1[4-7]_00.fits 
#
# This will run 4 parallel processes, working on
# files *_gpubox14_00.fits,  *_gpubox15_00.fits,
# *_gpubox16_00.fits, and *_gpubox17_00.fits.

# 
#

for dir in "$@"
do
    echo vv_fits -i "$dir" -f -t f32
    ~/bin/vv_fits -i "$dir" -f -t f32 &
done




















