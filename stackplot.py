from pylab import *
import numpy as np
import pandas
import pylab as plt


cell_f = np.loadtxt('662-689.cells', str)
cell_f = [ '/xfel/ffhs/dat/ue_180124/chufengl/indexing/taspase1_lcp_21/'+c for c in cell_f]

dfs = pandas.concat( [pandas.read_pickle(c) for c in cell_f])
run = list(map( lambda x: int( x.split("multi_dump/run")[1].split('_')[0]), dfs.cxi_fname)) 
runs = unique( run)
dfs['run'] = run
gb = dfs.groupby((dfs.run, dfs.lattice_type))
o = gb.count()


mono = o.iloc[ o.index.get_level_values("lattice_type")=='monoclinic']
tric = o.iloc[ o.index.get_level_values("lattice_type")=='triclinic']
hexa = o.iloc[ o.index.get_level_values("lattice_type")=='hexagonal']

M = mono[[ 'gamma']].reset_index()
H = hexa[[ 'gamma']].reset_index()
T = tric[[ 'gamma']].reset_index()

M.rename(columns={"gamma":"counts_mono"}, inplace=True)
H.rename(columns={"gamma":"counts_hexa"}, inplace=True)
T.rename(columns={"gamma":"counts_tric"}, inplace=True)

P = pandas.merge( pandas.merge( H,T,on='run', how='outer'  ), M, on='run', how='outer')
P['sum'] = P.counts_hexa+P.counts_mono+P.counts_tric
P = P.fillna(0)


figure(1)
fs=14
#plt.subplot(211)
#bar( P.run, P.counts_hexa/P['sum'], label='hexagonal')
#bar( P.run, P.counts_tric/P['sum'], bottom=P.counts_hexa/P['sum'], label='triclinic')
#bar( P.run, P.counts_mono/P['sum'], bottom=P.counts_hexa/P['sum']+P.counts_tric/P['sum'], label='monoclinic')
#ylabel("fraction", fontsize=fs) 
#legend()
#ylim(0,1.5)
#ax = gca()
#ax.set_xticklabels([])
#ax.tick_params(labelsize=fs)
#ax.set_xlim(661, 691)

bar( P.run, P.counts_hexa/100.+ P.counts_tric/100. + P.counts_mono/100.)#, label='hexagonal')
#bar( P.run, P.counts_tric/100., bottom=P.counts_hexa/100., label='triclinic')
#bar( P.run, P.counts_mono/100., bottom=P.counts_hexa/100.+P.counts_tric/100., 
#    label='monoclinic')
#legend()
ylabel("indexing rate (%) at 30 Hz", fontsize=fs)
xlabel("run", fontsize=fs)
ax = gca()
#prop={'size':14})
ax.tick_params(labelsize=fs)
hlines(0.1, 661, 691,color='C1')
ax.set_xlim(661, 691)
savefig("index_rate.png", dpi=150)
show()

