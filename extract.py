# This script extracts ROIs around manually annotated fibers, for use in later training.

import numpy as np
import skimage.measure as sm
import data
import math
import sys

tomogramfile = sys.argv[1]
annotationfile = sys.argv[2]
outputtomogram = sys.argv[3]
outputannotation = sys.argv[4]

# Read annotation and tomogram from disk
ann = data.read_tomogram(annotationfile, domean=False)
rec = data.read_tomogram(tomogramfile, domean=False)
print(ann.shape)
print(rec.shape)
print(ann.min(),ann.max())

# Determine bounds of annotated parts
mnsx = []
mnsy = []
mxsx = []
mxsy = []
zis = []
for i in range(ann.shape[0]):    
    if ann[i].max()>0:
        zis.append(i)
        cont = sm.find_contours(ann[i],0.05)
        for c in cont:
            mnsx.append(c[:,0].min())
            mnsy.append(c[:,1].min())
            mxsx.append(c[:,0].max())
            mxsy.append(c[:,1].max())
mnx = int(min(mnsx))
mny = int(min(mnsy))
mxx = int(math.ceil(max(mxsx)))
mxy = int(math.ceil(max(mxsy)))
print(mnx, mxx, mny, mxy)

# Extract annotated parts with padding
pd = 10
inp = rec[min(zis):max(zis)+1, mnx-pd:mxx+pd, mny-pd:mxy+pd]
tar = ann[min(zis):max(zis)+1, mnx-pd:mxx+pd, mny-pd:mxy+pd]
tar = (tar>0).astype(np.uint8)

# Save extracted ROI
import tifffile
tifffile.imsave(outputtomogram, inp)
tifffile.imsave(outputannotation, tar)
