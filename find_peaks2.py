import h5py
from skimage import measure 
import glob 
from joblib import Parallel, delayed
import numpy as np
from argparse import ArgumentParser
from scipy.signal import correlate2d
from scipy.ndimage.filters import maximum_filter, gaussian_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage import measurements
from scipy.spatial import cKDTree


def plot_pks( img, pk=None, **kwargs):
    if pk is None:
        pk,pk_I = pk_pos( img, **kwargs)
    m = img[ img > 0].mean()
    s = img[img > 0].std()
    imshow( img, vmax=m+5*s, vmin=m-s, cmap='viridis', aspect='equal', interpolation='nearest')
    ax = gca()
    for cent in pk:
        circ = plt.Circle(xy=(cent[1], cent[0]), radius=3, ec='r', fc='none',lw=1)
        ax.add_patch(circ)


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

def pk_pos( img_, make_sparse=False, nsigs=7, sig_G=None, thresh=1, sz=4, min_snr=2., 
    min_conn=2, filt=False):
    if make_sparse:
        img = img_[sz:-sz,sz:-sz].copy()
        m = img[ img > 0].mean()
        s = img[ img > 0].std()
        img[ img < m + nsigs*s] = 0
        if sig_G is not None:
            img = gaussian_filter( img, sig_G)
        lab_img, nlab = measurements.label(detect_peaks(gaussian_filter(img,sig_G)))
        locs = measurements.find_objects(lab_img)
        pos = [ ( int((y.start + y.stop) /2.), int((x.start+x.stop)/2.)) for y,x in locs ]
        pos =  [ p for p in pos if img[ p[0], p[1] ] > thresh]
        intens = [ img[ p[0], p[1]] for p in pos ] 
    else:
        img = img_[sz:-sz,sz:-sz].copy()
        if sig_G is not None:
            lab_img, nlab = measurements.label(detect_peaks(gaussian_filter(img,sig_G)))
        else:
            lab_img, nlab = measurements.label(detect_peaks(img))
        locs = measurements.find_objects(lab_img)
        pos = [ ( int((y.start + y.stop) /2.), int((x.start+x.stop)/2.)) for y,x in locs ]
        pos =  [ p for p in pos if img[ p[0], p[1] ] > thresh]
        intens = [ img[ p[0], p[1]] for p in pos ] 
    pos = np.array(pos)+sz
    
    if filt:
        new_pos = []
        new_intens =  []
        for (j,i),I in zip(pos, intens):
            im = img_[j-sz:j+sz,i-sz:i+sz]
            pts = im[ im > 0].ravel()
            bg = np.median( pts )
            if bg == 0:
                continue
            noise = np.std( pts-bg)
            #noise = np.median( np.sqrt(np.mean( (pts-bg)**2) ) )
            
            if I/bg < min_snr:
                continue

            im_n = im / noise
            blob = measure.label(im_n > min_snr)
            lab = blob[ sz, sz ]
            connectivity = np.sum( blob == lab)
            if connectivity < min_conn:
                continue
            new_pos.append( (j,i))
            new_intens.append( I)
        pos = new_pos
        intens = new_intens
    return pos, intens


def bin_ndarray(ndarray, new_shape):
        """
        Bins an ndarray in all axes based on the target shape, by summing or
            averaging.
        Number of output dimensions must match number of input dimensions.
        Example
        -------
        >>> m = np.arange(0,100,1).reshape((10,10))
        >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
        >>> print(n)
        [[ 22  30  38  46  54]
         [102 110 118 126 134]
         [182 190 198 206 214]
         [262 270 278 286 294]
         [342 350 358 366 374]]
        """
        compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                       ndarray.shape)]
        flattened = [l for p in compression_pairs for l in p]
        ndarray = ndarray.reshape(flattened)
        for i in range(len(new_shape)):
                ndarray = ndarray.sum(-1*(i+1))
        return ndarray


def make_cxi_file( hits, outname, Hsh, 
    mask=None,
    dtype=np.float32, 
    compression=None, 
    comp_opts=None, shuffle=False, thresh=1, sig_G=1, 
    make_sparse=1, nsigs=2, min_num_pks=0):
    all_pk = []
    all_pk_intens = []
    if mask is None:
        mask = np.ones(Hsh).astype(bool)
    with h5py.File( outname, "w") as out:
        #img_dset = out.create_dataset("images", 
        img_dset = out.create_dataset('data', 
            shape=(100000,Hsh[0], Hsh[1]),
            maxshape=(None,Hsh[0], Hsh[1] ), 
            dtype=dtype,
            chunks=(1,Hsh[0], Hsh[1]),
            compression=compression, 
            compression_opts=comp_opts,
            shuffle=shuffle)

        count = 0
        for h in hits: 
            pk, pk_I = pk_pos( h*mask, sig_G=sig_G, 
                thresh=thresh, 
                make_sparse=make_sparse, 
                nsigs=nsigs)
            npks = len(pk_I)
            if npks <= min_num_pks:
                continue
            print("Found %d peaks in hit.."%npks)
            all_pk.append( pk)
            all_pk_intens.append( pk_I)

            count += 1
            #img_dset.resize( (count, Hsh[0], Hsh[1]),)
            img_dset[count-1] = h
        img_dset.resize( (count, Hsh[0], Hsh[1]))
        npeaks = [len(p) for p in all_pk]
        max_n = max(npeaks)
        pk_x = np.zeros((len(all_pk), max_n))
        pk_y = np.zeros_like(pk_x)
        pk_I = np.zeros_like(pk_x)

        for i,pk in enumerate(all_pk):
            n = len( pk)
            pk_x[i,:n] = [p[1] for p in pk ]
            pk_y[i,:n] = [p[0] for p in pk ]
            pk_I[i,:n] = all_pk_intens[i]

        npeaks = np.array( npeaks, dtype=np.float32)
        pk_x = np.array( pk_x, dtype=np.float32)
        pk_y = np.array( pk_y, dtype=np.float32)
        pk_I = np.array( pk_I, dtype=np.float32)

        #out.create_dataset( 'entry_1/result_1/nPeaks' , data=npeaks)
        #out.create_dataset( 'entry_1/result_1/peakXPosRaw', data=pk_x )
        #out.create_dataset( 'entry_1/result_1/peakYPosRaw', data=pk_y )
        #out.create_dataset( 'entry_1/result_1/peakTotalIntensity', data=pk_I )
        out.create_dataset( 'peaks/nPeaks' , data=npeaks)
        out.create_dataset( 'peaks/peakXPosRaw', data=pk_x )
        out.create_dataset( 'peaks/peakYPosRaw', data=pk_y )
        out.create_dataset( 'peaks/peakTotalIntensity', data=pk_I )

    return all_pk, all_pk_intens

# blocks
def blocks(img, N):
    sub_imgs = []
    M = int( float(img.shape[0]) / N ) 
    y= 0
    for j in range( M):
        slab = img[y:y+N]
        x = 0
        for i in range(M):
            sub_imgs.append(slab[:,x:x+N] ) 
            x += N
        y += N
    return np.array( sub_imgs ) 

def make_temp(img, pk):
    temp = np.zeros_like( img)
    for i,j in pk:
        temp[i,j] = 1
    return temp

def make_temps(sub_imgs , pks ):
    temps = zeros_like ( sub_imgs ) 
    for i_pk, pk in enumerate(pks):
        for i,j in pk:
            temps[i_pk][i,j] = 1
    return temps
   

def make_2dcorr(img=None,  nsigs=4, sig_G=1.1, thresh=1, 
    make_sparse=1, block_sz=25, mode='full', temp=None ):
    if temp is None:
        assert (img is not None)
        pk,pk_I = pk_pos( img, sig_G=sig_G, 
            thresh=thresh, make_sparse=make_sparse, nsigs=nsigs)
        temp = make_temp( img, pk)
    
    temp_s = blocks( temp, N=block_sz)
    C = np.mean( [correlate2d(T,T, mode) for T in temp_s ], axis=0 ) 
    C[ C==C.max()] = 0 # mask self correlation
    return C







