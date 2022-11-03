import tifffile
import numpy as np
import msdnet
import sys
import data
import tqdm


# Class for applying test-time augmentation
class DataForApplication(object):
    def __init__(self, dats):
        self.dats = dats
    
    def augment(self, im, c):
        if c==1:
            return im[:,::-1]
        elif c==2:
            return im[:,:,::-1]
        elif c==3:
            return im[:,::-1,::-1]
        elif c==4:
            return np.rot90(im,1,axes=(1,2))
        elif c==5:
            return np.rot90(im,3,axes=(1,2))
        elif c==6:
            return np.rot90(im,1,axes=(1,2))[:,::-1]
        elif c==7:
            return np.rot90(im,3,axes=(1,2))[:,::-1]
    
    def augment_inv(self, im, c):
        if c==1:
            return im[:,::-1]
        elif c==2:
            return im[:,:,::-1]
        elif c==3:
            return im[:,::-1,::-1]
        elif c==4:
            return np.rot90(im,-1,axes=(1,2))
        elif c==5:
            return np.rot90(im,-3,axes=(1,2))
        elif c==6:
            return np.rot90(im[:,::-1],-1,axes=(1,2))
        elif c==7:
            return np.rot90(im[:,::-1],-3,axes=(1,2))
    
    def apply(self, i, n):
        if i<0 or i >= len(self.dats):
            raise ValueError("Out of range")
        im = (self.dats[i].input).copy()
        out = n.forward(im)
        for c in range(1,8):
            out += self.augment_inv(n.forward(self.augment(im, c).copy()), c)
        im = im[::-1].copy()
        out += n.forward(im)
        for c in range(1,8):
            out += self.augment_inv(n.forward(self.augment(im, c).copy()), c)
        return out

# Read tomogram from disk
a = data.read_tomogram(sys.argv[1], domean=False).astype(np.float32)

# Constant shift to pixel values (might be required to set to match with
# pixel value distribution of training tomograms)
shft = float(sys.argv[2])
a += shft
print(a.mean(), a.std(), a.min(), a.max())

# Set up data for test-time augmentation
dats = []
for i in range(a.shape[0]):
    d = msdnet.data.ArrayDataPoint(a[i:i+1])
    dats.append(d)
q = DataForApplication(msdnet.data.convert_to_slabs(dats, 2, flip=False))

# Load trained network
n = msdnet.network.SegmentationMSDNet.from_file('segm_params.h5', gpu=True)

# Apply network
out = np.zeros_like(a)
for i in tqdm.trange(out.shape[0]):
    out[i] = q.apply(i,n)[1]/16

# Save results to disk as 8-bit 3D image
out[out<0]=0
out[out>1]=1
out = (out*255).astype(np.uint8)
tifffile.imsave(sys.argv[3], out, compress=9)

