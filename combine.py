#!/usr/bin/python3
import h5py
import os
import glob
import sys
import numpy as np

tag = sys.argv[1]

indir = '/xfel/ffhs/dat/ue_180124/results/multi_dump'
outdir = '/xfel/ffhs/dat/ue_180124/results'

fs = os.path.join( indir, "%s*.cxi"%tag)
fnames = glob.glob(fs)

if not fnames:
    print("no fnames for tag %s"%tag)
    sys.exit()

keys = [ '/data',
  '/peaks/nPeaks',
  '/peaks/peakTotalIntensity',
  '/peaks/peakXPosRaw',
  '/peaks/peakYPosRaw',
  ]


good_fnames = []
for f in fnames:
    try:
        h = h5py.File(f, 'r')
        for k in keys:
            assert( h[k].shape[0] > 0 )
        good_fnames.append( f ) 
    except:
        pass

if not good_fnames:
    print("no good fnames for tag %s"%tag)
    sys.exit()

n = 0
for f in good_fnames:
    n = max( n, h5py.File(f,'r')['peaks/nPeaks'].value.max())
n = int(n)
print ("max nPeaks: %d"%n) 

shapes = { "/data":[None, 1440, 1440],
 "/peaks/nPeaks":[None],
 "/peaks/peakXPosRaw":[None, n],
 "/peaks/peakYPosRaw":[None, n],
 "/peaks/peakTotalIntensity":[None, n] }

outname = os.path.join( outdir,  "%s_ALL.cxi"%tag)

with h5py.File(outname, "w") as out:
    dsets = {}
    dset_size = { k:0 for k in keys}

    f = good_fnames[0]
    h5 = h5py.File(f,'r')
    for k in keys:
        data = h5[k].value
        if k in ['/peaks/peakYPosRaw', 
                '/peaks/peakXPosRaw', '/peaks/peakTotalIntensity'] :
            n_old = data.shape[1]
            data2 = np.zeros( ( data.shape[0], n) )
            data2[:,:n_old] = data
            data = data2.copy()
        sh = shapes[k]
        dsets[k] = out.create_dataset( k, data=data, 
            compression=None, 
            maxshape = sh)
        dset_size[k] += data.shape[0]
       
    if not good_fnames[1:]:
        sys.exit()
    for f in good_fnames[1:]:
        h5 = h5py.File(f, 'r')
        print( f)
        for k in keys:
            data = h5[k].value
            if data.shape[0]==0:
                break            
            
            if k in ['/peaks/peakYPosRaw', 
                    '/peaks/peakXPosRaw', '/peaks/peakTotalIntensity'] :
                n_old = data.shape[1]
                data2 = np.zeros( ( data.shape[0], n) )
                data2[:,:n_old] = data
                data = data2.copy()
            
            sh = list(shapes[k]) 
            start = dset_size[k]
            stop = data.shape[0] + start
            sh[0] = n + start
            dsets[k].resize( sh )
            dsets[k][start:stop ] = data
            dset_size[k] += data.shape[0]

