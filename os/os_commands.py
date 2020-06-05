import os

user = os.path.expanduser('~')
#                                                                    
# find ~ -type d -name "casa-release*"
#                                                                       
casa_dirs = []
for root, dirs, files in os.walk(user):
    for d in dirs:
        if d.find('casa-release') <> -1:
            casa_dirs.append(root + '/' + d)

            print 'CASA release found at %s' % casa_dirs[-1]
