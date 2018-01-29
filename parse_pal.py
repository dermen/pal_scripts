
import pylab as plt
import h5py
import sys
import joblib
from find_peaks import make_cxi_file

h5 = h5py.File( sys.argv[1], 'r')
keys = list( h5.keys() )[1:]
imgs = [ h5[k]['data'] for k in keys ]

make_cxi_file( imgs, 
	outname = sys.argv[2],
	Hsh = (1440,1440),	
	sig_G=1, 
	thresh=10, 
	make_sparse=True,
	nsigs=3, min_num_pks=10)








