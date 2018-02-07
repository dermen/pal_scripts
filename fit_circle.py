# coding: utf-8
import h5py
f = h5py.File('HDF5_MX_0000006__003/MX_0000006_0000052.h5','r')
f['data']
f['data/data']
f['data']
img = f['data'].value
imshow( img, vmax=199)
get_ipython().run_line_magic('pwd', '')
s = '/xfel/dcs1/dat/commission/cm_20170408/proc/201704-all'
get_ipython().run_line_magic('cd', '')
cls
get_ipython().run_line_magic('cd', '/xfel/ffhs/')
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('cd', 'dat/')
get_ipython().run_line_magic('cd', 'ue_180127/')
get_ipython().run_line_magic('cd', 'bin')
from find_peaks4 import pk_pos
from find_peaks4 import plot_pks
plot_pks( img, nsigs=4, filt=False, thresh=50, sig_G=1.1, make_sparse=True)
pk,_ = pk_pos( img, nsigs=4, filt=False, thresh=50, sig_G=1.1, make_sparse=True)
pk,_ = pk_pos( img, nsigs=4 thresh=50, sig_G=1.1, make_sparse=True)
pk,_ = pk_pos( img, nsigs=4 ,thresh=50, sig_G=1.1, make_sparse=True)
pk
y,x = list(map(array, list(zip(*pk))))
y
x
for j,i in zip(y,x):
    img[j-3:j+3, i-3:i+3] = 0
    
cla()
imshow( img, vmax=199)
import sys
sys.append("../.asu_tools/lib/python")
sys.path.append("../.asu_tools/lib/python")
from loki.RingData import RingFit
rf = RingFit(img)
#xi,yi = (
img.shape
xi,yi = (1440,1440)
get_ipython().run_line_magic('pinfo', 'rf.fit_circle_fast')
#rf.fit_circle_fast( (xi,yi, )
xi-1137
ri = 303
rf.fit_circle_fast( (xi,yi,ri ), num_fitting_pts=5000, num_high_pix=100, ring_width=100)
x,y,r = array([ 1450.10619788,  1430.22695615,   316.51128542])
from loki.RingData import RadialProfile
rp = RadialProfile( (xi,yi), img.shape)
figure(2);plot( rp.calculate(img))
rp = RadialProfile( (x,y), img.shape)
figure(2);plot( rp.calculate(img))
#rf.fit_circle_fast( (xi,yi,ri ), num_fitting_pts=5000, num_high_pix=100, ring_width=100)
circ = Circle(xy=(x,y), radius=r, fc='none', ec='r')
figure(1);gca().add_patch(circ);draw()
rf.fit_circle_fast( (xi,yi,ri ), num_fitting_pts=10000, num_high_pix=100, ring_width=150)
xmymr
x,y
rf.fit_circle_slow?#( (xi,yi,ri ),)
get_ipython().run_line_magic('pinfo', 'rf.fit_circle_slow')
get_ipython().run_line_magic('pinfo', 'rf.fit_circle_slow')
rf.fit_circle_slow( (x,y,r ), 50,30,2)
get_ipython().run_line_magic('pinfo', 'rf.fit_circle_slow')
rf.fit_circle_slow( (x,y,r ),  50.,30.,2.)
rf.fit_circle_slow( (x,y,r ),  50,30)
#rf.fit_circle_slow( (x,y,r ),  50,30)
x
y
get_ipython().run_line_magic('save', '1-65 fit_circle')
