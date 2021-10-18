#%%
#! /usr/bin/env python
# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2019, Acoular Development Team.
#------------------------------------------------------------------------------
"""Demo for Acoular

Generates a test data set for three sources, analyzes them and generates a
map of the three sources.
 
The simulation generates the sound pressure at 64 microphones that are
arrangend in the 'array64' geometry, which is part of the package. The sound
pressure signals are sampled at 51200 Hz for a duration of 1 second.

Source location (relative to array center) and levels:

====== =============== ======
Source Location        Level 
====== =============== ======
1      (-0.1,-0.1,0.3) 1.0 Pa
2      (0.15,0,0.3)    0.7 Pa 
3      (0,0.1,0.3)     0.5 Pa
====== =============== ======
"""

from os import path
from acoular import __file__ as bpath, config, MicGeom, WNoiseGenerator, PointSource,\
 Mixer, WriteH5, TimeSamples, PowerSpectra, RectGrid, SteeringVector,\
 BeamformerBase, L_p, BeamformerEig, BeamformerOrth, BeamformerCleansc
from pylab import figure, plot, axis, imshow, colorbar, show
from _spectra_import import PowerSpectraImport
from _jahnke_reverse import *
from main import train_data,A
from test_models import reshape_sourcemap
from numpy import expand_dims, zeros, matmul
from loss_funcs import nmse_db
from matplotlib import pyplot as plt

config.global_caching="none" # disable caching!

# set up the parameters
sfreq = 40*343.
duration = 1
nsamples = duration*sfreq
micgeofile = path.join(path.split(bpath)[0],'xml','array_64.xml')
micgeofile = "_tub_vogel16_ap1.xml"
h5savefile = 'three_sources.h5'

# generate test data, in real life this would come from an array measurement
mg = MicGeom( from_file=micgeofile )
n1 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=1 )
n2 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=2, rms=0.7 )
n3 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=3, rms=0.5 )
p1 = PointSource( signal=n1, mics=mg,  loc=(-0.1,-0.1,0.3) )
p2 = PointSource( signal=n2, mics=mg,  loc=(0.15,0,0.3) )
p3 = PointSource( signal=n3, mics=mg,  loc=(0,0.1,0.3) )
pa = Mixer( source=p1, sources=[p2,p3] )
# wh5 = WriteH5( source=pa, name=h5savefile )
# wh5.save()

# set the CSM of the PowerSpectra object at PowerSpectraImport
ps = PowerSpectra( time_data=pa, block_size=128, window='Hanning' )
# csm = ps.csm[:,:,:].copy() # copy the csm from PowerSpectra object
y,x = next(iter(train_data))
y_simu = y.numpy()[0]
y_calc = matmul(A,x[0].numpy())
sourcemaps=[]
for y_ in [y_simu,y_calc ]:
	y_ = unstack_complex_vector(y_)
	csm_ = transform_y_to_csm(y_)
	csm_ = expand_dims(csm_,axis=0)
	csm = zeros((65,16,16),dtype=complex)
	csm[48] = csm_


	ps_import = PowerSpectraImport(csm=csm, sample_freq=sfreq) # it is mandatory to also set the sample_freq attribute!

	# analyze the data and generate map
	# ts = TimeSamples( name=h5savefile )
	# ps = PowerSpectra( time_data=ts, block_size=128, window='Hanning' )

	rg = RectGrid( x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z=.5,increment=0.04 )
	st = SteeringVector(grid=rg, mics=mg)

	bb = BeamformerBase( freq_data=ps_import, steer=st, r_diag=True)
	pm = bb.synthetic( 15*343., 0 )
	Lm = L_p( pm ).T
	sourcemaps.append(Lm)


# fig = plt.figure(figsize=(16,9))
# ax1 = fig.add_subplot(131)
# ax1.imshow(sourcemaps[0])
# ax1.set_title("Simulated")
# ax2 = fig.add_subplot(132)
# ax2.imshow(sourcemaps[1])
# ax2.set_title("Calculated")
# ax3 = fig.add_subplot(133)
# ax3.imshow(reshape_sourcemap(x[0].numpy()))
# ax3.set_title("True")
# plt.colorbar(ax1)
# fig.show()

plt.figure()
plt.imshow(reshape_sourcemap(x[0].numpy()))
plt.colorbar()
plt.figure()
plt.imshow(sourcemaps[1],origin="lower",vmax=95,vmin=75)
plt.title("Calculated")
plt.colorbar()
plt.figure()
plt.imshow(sourcemaps[0],origin="lower",vmax=95,vmin=75)
plt.title("Simulated")
plt.colorbar()


#%%

i = find_indices(mg)
csms = [None,None]
csm = ps.csm[13]
csms[0] = csm
csms[1] = transform_y_to_csm(((transform_csm_to_y(csm,i))))

fig,ax = plt.subplots(1,3,sharey=True)
ax[0].imshow(csms[0].real)
ax[1].imshow(csms[1].real)
ax[2].imshow((csms[0].real - csms[1].real)**2)

#%%

# show map
imshow( Lm.T, origin='lower', vmin=Lm.max()-10, extent=rg.extend(), )
colorbar()

# plot microphone geometry
figure(2)
plot(mg.mpos[0],mg.mpos[1],'o')
axis('equal')

show()

# %%
