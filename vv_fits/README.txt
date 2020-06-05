One can do the van Vleck coreection of GPU fits files in several steps:

1. Getting the software

1.1. The easiest way is to copy (-recursively!) to your server the directory
vv_fits/ with all the files needed from jansky to <where>: 

scp -r jansky.haystack.mit.edu:/data/vv_fits your.server.name:~/<where>

1.2. Alternatively, you can checkout this directory from the repository:

svn checkout svn+ssh://mwa-lfd.haystack.mit.edu/svn/MISC/vv_fits <some_dir>


2. Building

For more detailed instructions please look into the comments making the header
of vv_fits/vv_fits.c file. Use this line to compile and build the vv_fits
executable: 

gcc -g  -lcfitsio -lm mwac_utils.c antenna_mapping.c \
         vanvleckf.c   vv_fits.c   -o vv_fits

3. The vv-correction proper.

Again, you can find some details in the vv_fits.c header and in its help output.

Use the command line like this (add -v for verbosity (loquacity? :)  

$ ./vv_fits -i 1130642936_20151104032841_gpubox11_00.fits -f

This command will take ~1.5 minutes and it will generate the vv-corrected fits
file 1130642936_20151104032841_vv_gpubox11_00.fits .

Remember: the directory where you run vv_fits must contain 4 (pretty voluminous)
binary tables:
qsigf.bin  
rho_rulerf.bin  
sig_rulerf.bin  
xy_tablef.bin

The suffices "f" at their names indicate that the data there are in single
precision, 32-bit floats, not 64-bit doubles.

3.1. You can significantly accelerate the vvc using the script "vv_fits.bat".

Run it with a wildcard, like 

$ bash vv_fits.bat *_gpubox1[4-7]_00.fits 

This will run 4 parallel processes, working on files *_gpubox14_00.fits,
*_gpubox15_00.fits, *_gpubox16_00.fits, and *_gpubox17_00.fits. 

With my 8 CPUs and just 8 GB of RAM I could run 10 processes at ones without
danger of employing the swap memory. Like the following:

$ bash vv_fits.bat *_gpubox0?_00.fits

The script started simultaneous work on files from *_00_00.fits to *_09_00.fits.

So I created 24 vv_corrected gpu fits files in three moves.

4. Making CASA MS using cotter

I can provide here the command I used and which was successful.

cotter -o 1130642936_20151104032841_vv_.ms -timeres 0.5 -freqres 40 -m
../1130642936_metafits_ppds.fits -norfi -nostats -nosbgains -allowmissing
-flagdcchannels -usepcentre -edgewidth 0 -initflag 0 -mem 80
../*_vv_gpubox??_??.fits 

Some comments:

*_vv_gpubox??_??.fits -- the template for all 24 vv-corrected fits files.

-mem 80 -- use less than 80% of memory, to be able to do something else in
parallel.

1130642936_20151104032841_vv_.ms -- should contain _vv_ or any other sign to
distinguish the vvc MSes.


4.1. There is one caveat. The metafits file 1130642936_metafits_ppds.fits
assumes there are 4 times 24 files, but we use only 1 24-file set. So one keword
in the FIRST header of the meta file must be changed before running cotter. Here
I provide what John Morgan wrote to us:

"In addition to

1130642936_20151104032841_gpubox01_00.fits

The gpubox data for the full observation will also include

1130642936_20151104032941_gpubox01_01.fits
1130642936_20151104033041_gpubox01_02.fits
1130642936_20151104033141_gpubox01_03.fits

There is typically one gpubox file per coarse channel, per minute of 
observation.

Cotter should handle with this gracefully and produce a valid 
measurement set. Note however that the missing data is only "flagged". I 
believe that this means that the measurement set will be almost 4 times 
bigger than it needs to be (containing mostly zeros).

To resolve this issue, you can change the value of NSCANS to 120 in the 
metafits file."

I used IPython to fix this. 
$ ipython --pylab
[1] import pyfits
[2] hl = pyfits.open('1130642936_metafits_ppds.fits', mode='update')
[3] h0 = hl[0]
[4] h0.header['NSCANS'] = 120
[5] hl.close()

Good luck!

