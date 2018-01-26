
import numpy as np
import pylab as plt
from astride import Streak
import h5py


# load some images
imgs = np.load('streak_igms_PINK.h5py.npy' )
x,y,I = np.load('streak_peaks_PINK.h5py.npy')
inds = [46, 49, 50, 53]
all_pk = []
for i in inds:
    all_pk.append( map( np.array , zip( y[i], x[i] ) ) )
pk = all_pk[0]
subs = make_sub_imgs( imgs[0], pk, 10 )

img = subs[10]
img_sh = img.shape

# define coordinates of each pixel for checking later
Y,X = np.indices( img_sh)
pix_pts = np.array( zip(X.ravel(), Y.ravel() ) )

def mask_streak( img):
#   make the streak detector
    streak = Streak(img, output_path='.')
streak.detect()
 
#   make a mask from the streaks
edges = streak.streaks

#   these vertices define polygons surrounding each streak
verts = [ np.vstack(( edge['x'], edge['y'])).T 
    for edge in edges]

#   make a path object corresponging to each polygon
paths = [ mpl.path.Path(v) 
    for v in verts ]

#   check if pixel is contained in ANY of the streak polygons
    contains = np.vstack( [ p.contains_points(pix_pts) 
        for p in paths ])
    
    mask = np.any( contains,0).reshape( img_sh)

    return np.logical_not(mask)

fig, ax = plt.subplots( 1,2, figsize=(12,12))




########
# process

import numpy as np
import pylab as plt
from astride import Streak
import h5py
from itertools import izip
from scipy.ndimage.filters import gaussian_filter

def make_sub_imgs( img, pk, sz):
    ymax = img.shape[0]
    xmax = img.shape[1]

    y,x = map(np.array, zip(*pk))

    sub_imgs = []
    for i,j in izip( x,y):
        j2 = int( min( ymax,j+sz  ) )
        j1 = int( max( 0,j-sz  ) )
        
        i2 = int( min( xmax,i+sz  ) )
        i1 = int( max( 0,i-sz  ) )
        sub_imgs.append( img[ j1:j2, i1:i2 ] )
 
    return sub_imgs


# load some images
imgs = np.load('streak_igms_PINK.h5py.npy' )
x,y,I = np.load('streak_peaks_PINK.h5py.npy')
inds = [46, 49, 50, 53]
all_pk = []
for i in inds:
    all_pk.append( map( np.array , zip( y[i], x[i] ) ) )
pk = all_pk[0]
subs = make_sub_imgs( imgs[0], pk, 10 )
# len(subs) == 44

nrows=6
ncols=7
fig,axs = subplots(nrows=nrows, ncols=ncols)
k = 0
for row in xrange(nrows):
    for col in xrange( ncols):
        img = subs[k]
        ax = axs[row,col]
        ax.clear()
        ax.set_title(str(k))
        ax.imshow(img, aspect='auto')
       
#       raw version
        streak = Streak(gaussian_filter(img,0), output_path='.', 
            min_points=0, 
            shape_cut=1, 
            area_cut=10, 
            radius_dev_cut=0., 
            connectivity_angle=10.)
        streak.detect()
        edges = streak.streaks
        if edges:
            verts = [ np.vstack(( edge['x'], edge['y'])).T 
                for edge in edges]
            paths = [ mpl.path.Path(v) 
                for v in verts ]
            for p in  paths:
                patch = mpl.patches.PathPatch(p, facecolor='none', lw=2, edgecolor='Deeppink')
                ax.add_patch(patch)

#       smoothed version
        streak = Streak(gaussian_filter(img,1.4), output_path='.', 
            min_points=0, 
            shape_cut=1, 
            area_cut=10, 
            radius_dev_cut=0., 
            connectivity_angle=10.)
        streak.detect()
        edges = streak.streaks
        if edges:
            verts = [ np.vstack(( edge['x'], edge['y'])).T 
                for edge in edges]
            paths = [ mpl.path.Path(v) 
                for v in verts ]
            for p in  paths:
                patch = mpl.patches.PathPatch(p, facecolor='none', lw=2, edgecolor='w')
                ax.add_patch(patch)

        ax.set_xticks([])
        ax.set_yticks([])
        k+=1



