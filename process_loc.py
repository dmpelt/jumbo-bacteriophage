# This script extracts ROIs around manually annotated phage locations, and rotates them all in the same orientation.

import tifffile
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys
import glob
import data

# Input folder with tomograms (MRC), network annotations (tiff), and manual location annotations (txt)
dr = sys.argv[1]

# Read tomogram and network annotation
tomo = glob.glob(dr +'/*.mrc')[0]
nn = glob.glob(dr +'/*.tiff')[0]
c = tifffile.imread(nn)
fc = RegularGridInterpolator((range(c.shape[0]), range(c.shape[1]), range(c.shape[2])), c, bounds_error=False, fill_value=0)
a = data.read_tomogram(tomo, domean=False).astype(np.float32)
fa = RegularGridInterpolator((range(a.shape[0]), range(a.shape[1]), range(a.shape[2])), a, bounds_error=False, fill_value=0)

# List of manual annotation files
txts = glob.glob(dr +'/*.txt')
print(txts)

# For all annotated locations
for t in txts:
    # Load manual locations
    b = np.loadtxt(t)
    # For all annotations in the file, extract ROI around annotated phage,
    # and rotate tomogram and annotation to have the phage vertically.
    for i in range(0, b.shape[0], 2):
        diffvec = np.array((b[i+1][2] - b[i][2], b[i+1][1] - b[i][1], b[i+1][0] - b[i][0]))
        bas1 = diffvec/np.sqrt(np.sum(diffvec**2))
        if bas1[0]!=0:
            bas2 = np.cross(bas1, [1,0,0])
        elif bas1[1]!=0:
            bas2 = np.cross(bas1, [0,1,0])
        else:
            bas2 = np.cross(bas1, [0,0,1])
        bas2 = bas2/np.sqrt(np.sum(bas2**2))
        bas3 = np.cross(bas1, bas2)
        bas3 = bas3/np.sqrt(np.sum(bas3**2))
        
        zz, yy, xx = np.mgrid[-100:101, -100:101, -100:101]

        coor = np.zeros((201, 201, 201, 3))
        coor[...,0] = zz*bas1[0] + yy*bas2[0] + xx*bas3[0] + b[i][2]
        coor[...,1] = zz*bas1[1] + yy*bas2[1] + xx*bas3[1] + b[i][1]
        coor[...,2] = zz*bas1[2] + yy*bas2[2] + xx*bas3[2] + b[i][0]

        inpa = fa(coor.reshape((-1, 3))).reshape((201,201,201)).astype(np.float32)
        inpc = fc(coor.reshape((-1, 3))).reshape((201,201,201))
        print(inpc.min(),inpc.max())
        inpc[inpc<0]=0
        inpc[inpc>255]=255

        # Save to disk
        tifffile.imsave(t[:-4]+'_tomo{:02d}.tiff'.format(i//2+1), inpa.swapaxes(0,2).swapaxes(1,2)[:,::-1])
        tifffile.imsave(t[:-4]+'_fiber{:02d}.tiff'.format(i//2+1), inpc.swapaxes(0,2).swapaxes(1,2)[:,::-1].astype(np.uint8))
    



    
