'''
find_unique_photos.py

Finds unique file names with the specified extensions
in the directory tree starting with topdir. 

'''
import os, sys, glob
import numpy as np

# defaultenc = sys.getdefaultencoding()
# filesysenc = sys.getfilesystemencoding()

#topdir = 'C:\\Users\\benkev'
#topdir = 'D:\\Downloads'
#topdir = 'D:\\Users\\benkev\\Ph'
#topdir = 'D:\\Users\\benkev'
#topdir = 'F:\\Downloads'
#topdir = 'C:\\Users\\benkev\\Documents\\mp4'

topdir = 'E:\\'
#topdir = 'C:\\'
excld = ['Windows', 'Windows.old', 'Program Files', 'Program Files (x86)', \
         'ProgramData', 'PerfLogs', '$Recycle.Bin', 'Boot', 'Intel', \
         '$AV_AVG', '$WINDOWS.~BT', '$Windows.~WS', 'PerfLogs']

#
# Form output file name fout_name
#
pname = topdir.replace(':', '').replace('\\', '_')
fout_name = 'photos_found_in_' + pname + '.utf8.txt'
if fout_name[-1] == '_': fout_name = fout_name[:-1] # Cut trailing '_', if any

#
# Rule out the unneeded directories
#
if topdir[-1] != '\\': topdir += '\\'
# Prepend the excluded dirs in the excld with topdir
excldir = [topdir+excl for excl in excld]
topdir_content = glob.glob(topdir + '*')
topdirs = []           # Select only directories from all the content
for ent in topdir_content:
    if os.path.isdir(ent):
        if ent not in excldir:
            topdirs.append(ent)



fpict = []        # The pictures file names found without paths
fpath = []        # The paths to the picture files found
fpath_pict = []   # The pictures files with full paths found 
ifile = 0

for dir in topdirs:
    for path, dirs, files in os.walk(dir): 
        print('path = "%s"' % path)
        for file in files:       # file runs over all files in 'path' dir
            fname, ext = os.path.splitext(file)
            if ext in ('.jpg', '.jpeg'):
                fpict.append(file)
                fpath.append(path)
                fpath_pict.append(path + '\\' + file)
                ifile = ifile + 1

apict = np.array(fpict)
apath = np.array(fpath)
apath_pict = np.array(fpath_pict)
upict, ixu = np.unique(apict, return_index=True)
upath = apath[ixu]
upath_pict = apath_pict[ixu]

#
# Find duplicates and their locations 
#
nupict = len(upict)  # Number of unique pictures
#dpath_pict = []           # Duplicate pictures with full paths
#dpath = []           # Paths of the duplicate pictures
# Duplicate pictures dictionary <pict>:array(idx_path1, idx_path2, ...]
dpict = {}

for iupict in range(nupict):
    ixdup = np.where(upict[iupict] == apict)[0] # Indices of dups into apath
    if len(ixdup) > 1:  # If iupict-th picture is in several directories,
        dpict[upict[iupict]] = ixdup # put it in dict as a key:ixdup


with open(fout_name, 'w', encoding="utf-8") as fh:
    fh.write('#\n# Photos found in directories under '+topdir+'\n')
    fh.write('#\n')
    fh.write('# Excluded from search:\n')
    #fh.write('#\n')
    for excl in excldir:
        fh.write('#    ' + excl+'\n')
    fh.write('#\n\n')
    for pic, ixdirs in dpict.items():
        fh.write(pic+'\n')
        for ixdir in ixdirs:
            fh.write('  ' + apath[ixdir]+'\n')
        fh.write('\n')

        











