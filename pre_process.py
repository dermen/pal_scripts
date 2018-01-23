"""
Use this script as template for processing the data files stored on disk
"""

import h5py
import fabio
import glob 
from joblib import Parallel, delayed
import numpy as np
from argparse import ArgumentParser

from peak_utils import pk_pos, bin_ndarray

# arguments...
parser = ArgumentParser(
    description='open MCCD image files, check for peaks, then save hit filenames in current directory')

parser.add_argument(
    '-p',
    dest='prefix',
    type=str,
    default=None,
    help="file name prefix, files will be named according to this prefix and the number of jobs used")

parser.add_argument(
    '-i',
    dest='fnames',
    type=str,
    default=None,
    help="input mccd filenames list")

parser.add_argument(
    '-j',
    dest='n_jobs',
    type=int,
    default=1, help='number of jobs to run in parallel')
args = parser.parse_args()


# binned image shape (work this out ahead of time
bin_sh = (1280,1280)

# radius cutoff for hit detection, require certain number of peaks within this radius from center of image
# I found hit detection was easier using this portion of the image for A2a, not necessary
# for data with many peaks e.g. phyco, in which case you can change code accordingly.
max_rad = 350

# peak detection params, see function pk_pos in peak_utils
sig_G = 1.1 # 
thresh=1 # 
make_sparse=True # 
nsigs=7 # 

# min number of peaks to be considered a hit
min_npeaks = 5
#====================

def dump_filenames(fnames, jid):
#   this is the filename for output
    output_name ="%s_%d"%(args.prefix, jid)
    Nshots = len( fnames )  # number of filename, single images per MCCD file
    hit_fnames = [] # this is a list that will contain the file names correspnding to hits
    
    for counter,f in enumerate(fnames):
        print("\tJOB %d\t shot %d / %d"%(jid, counter+1, Nshots ) )
        try:
            img = fabio.open(f).data # use fabio library to read the file, extract the pixels
        except:
            print("\tJOB %s\tfabio could not open the file %s"%(jid, f ))
            continue
        # down-sample the image, this should have bin tested ahead of time, such that bin_sh works without error, should be multiple of the img size
        img = bin_ndarray(img, bin_sh) 
        # here I mask th beamstop, again, work this out ahead of time and change numbers accordingly
        img[ 630:710, 590:710] = 0
        
#       returns list of peak positions [ (y0,x0), (y1,x1) ... ]
#       as well as list of peak intensities [ I0, I1, I2  ... ]
        pkYX, pkI = pk_pos( img, 
            sig_G=sig_G, 
            thresh=thresh, 
            make_sparse=make_sparse, 
            nsigs=nsigs)
        
        if not pkYX:
            print("\tJOB %d\tNo detected peaks in image file %s"%(jid, f))
            continue
#       y corresponds to slow-scan, x to fast-scan coordinates
        y,x = map(np.array, zip( * pkYX) )
        
#       this is approximate peak radius, because we have not optimized the center position.. 
        r = np.sqrt(  (x-bin_sh[1]/2.)**2 + (y-bin_sh[0]/2.)**2)
#       compute number of peaks, keeping track of the peaks within the max radius
        npeaks = len([ pk for i,pk in enumerate(pkYX) 
            if r[i] < max_rad ])

        if npeaks >= min_npeaks:
            print("HIT!")
            hit_fnames.append(f)
    np.save(output_name, hit_fnames)


# run the above function across n_jobs
fnames = np.loadtxt( args.fnames, str)
results = Parallel(n_jobs=args.n_jobs)(delayed(dump_filenames)(fs,jid ) 
    for jid, fs in enumerate(np.array_split( fnames, args.n_jobs) ))

