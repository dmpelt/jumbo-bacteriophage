# This script provides a function to read MRC files.

import numpy as np

def read_tomogram(fn, domean=True):    
    # Reads tomogram from MRC file
    try:
            fid = open(fn, 'rb')
    except:
        raise Exception('Error opening file')
    dims = np.fromfile(fid, dtype=np.dtype('u4'), count=3)
    type = np.fromfile(fid, dtype=np.dtype('u4'), count=1)[0]
    fid.seek(23 * 4)
    next = np.fromfile(fid, dtype=np.dtype('u4'), count=1)[0]
    fid.seek(1024 + next)
    rec = np.fromfile(fid, dtype=np.dtype('f4'), count=dims[0]*dims[1]*dims[2]).reshape((dims[2],dims[1],dims[0]))
    fid.close()
    if domean:
        rec -= rec.mean()
        rec /= rec.std()
    return rec
