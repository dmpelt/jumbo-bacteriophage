# Import code
import msdnet
import tifffile

# Initialize network
dilations = msdnet.dilations.IncrementDilations(5)
n = msdnet.network.SegmentationMSDNet(100, dilations, 5, 2, gpu=True, softmaxderiv=False)
n.initialize()

# Define training data
dats_augm = []
datsv = []
for i in range(1,8):
    # Read input tomogram and target annotation
    a = tifffile.imread('inp{}.tiff'.format(i))
    b = tifffile.imread('tar{}.tiff'.format(i))

    dats = []
    for i in range(a.shape[0]):
        d = msdnet.data.ArrayDataPoint(a[i:i+1],b[i:i+1])
        d_oh = msdnet.data.OneHotDataPoint(d, [0,1])
        dats.append(d_oh)
    dats_augm.extend([msdnet.data.RotateAndFlipDataPoint(d) for d in msdnet.data.convert_to_slabs(dats, 2, flip=True)])
    datsv.extend(msdnet.data.convert_to_slabs(dats, 2, flip=False))
    print(len(dats_augm), len(datsv))

# Also add background images to training set
# Random crops of 150x150 pixels will be used, matching the rough
# size of an annotated image part.
import data
import numpy as np
import glob
tararr = np.zeros((2, 150, 150), dtype=np.float32)
tararr[0]=1
class RandomCropData(msdnet.data.DataPoint):
    def __init__(self, fn, fix=False) -> None:
        self.arr = data.read_tomogram(fn, domean=False)
        self.shp = self.arr.shape
        self.fix = fix
        if fix:
            zidx = int(np.random.random()*(self.shp[0]-5))
            yidx = int(np.random.random()*(self.shp[1]-150))
            xidx = int(np.random.random()*(self.shp[2]-150))
            self.arr = self.arr[zidx:zidx+5, yidx:yidx+150, xidx:xidx+150].copy()

    
    def getinputarray(self):
        if self.fix:
            return self.arr.copy()
        else:
            zidx = int(np.random.random()*(self.shp[0]-5))
            yidx = int(np.random.random()*(self.shp[1]-150))
            xidx = int(np.random.random()*(self.shp[2]-150))
            return self.arr[zidx:zidx+5, yidx:yidx+150, xidx:xidx+150].copy()
    
    def gettargetarray(self):
        return tararr.copy()
for f in glob.glob('backgrounds/*.mrc'):
    for i in range(80):
        dats_augm.append(msdnet.data.RotateAndFlipDataPoint(RandomCropData(f)))
        datsv.append(RandomCropData(f,fix=True))

print(len(dats_augm), len(datsv))

# Use batches of 10 images
bprov = msdnet.data.BatchProvider(dats_augm,10)

# Normalize input and target
n.normalizeinout(datsv)

# Set up training
celoss = msdnet.loss.L2Loss()
val = msdnet.validate.LossValidation(datsv, loss=celoss)
t = msdnet.train.AdamAlgorithm(n, loss=celoss)
consolelog = msdnet.loggers.ConsoleLogger()
filelog = msdnet.loggers.FileLogger('log.txt')
imagelog = msdnet.loggers.ImageLabelLogger('log', chan_in=2, onlyifbetter=True)
singlechannellog = msdnet.loggers.ImageLogger('log_singlechannel', chan_in=2, chan_out=1, onlyifbetter=True)
msdnet.train.train(n, t, val, bprov, 'segm_params.h5',loggers=[consolelog,filelog,imagelog,singlechannellog], val_every=len(datsv)//10, progress=True)
