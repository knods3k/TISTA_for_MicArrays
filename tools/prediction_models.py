#%%
import tools
#%%
from os.path import normpath, join
from acoular import config, L_p, BeamformerCleansc, BeamformerCMF
from tools.scratchfiles._spectra_import import PowerSpectraImport
from tools.scratchfiles._jahnke_reverse import *
import numpy as np
from tensorflow.keras.models import load_model

from tools.environment import  MODELPATH
from tools.physical import PhyiscalModel

from scipy.signal import convolve2d

config.global_caching="none" # disable caching


def magnify(input, scale=3):
	m = scale
	res = 99*2
	x = np.linspace(-res,res,scale)
	y = x[:,None]
	r = x**2 + y**2
	circle = r < (scale/3)**2
	return convolve2d(input, circle, mode="same", boundary="symm")

def bin(input, scale=2):
	uneven = input.shape[0]%2 != 0
	if uneven:
		input = np.pad(input, ((0,1),(0,1)), 'constant') 
	N = input.shape[0]
	input = input.reshape(N//scale, scale, N//scale, scale)
	input = input.sum(axis=(1,3))
	
	return input

def tista(y_, physics, T=60):
	he = physics.he
	path = normpath(join(MODELPATH, f'He={he}', f'He={he}_T={T}'))
	model = load_model(path)
	x_pred = model(y_)
	x_pred = physics.vector_to_sourcemap(x_pred)
	return x_pred.numpy()


def cleansc(y_, physics):
	freq = physics.freq
	sfreq = 10*freq

	y_ = physics.unstack_complex_vector(y_).numpy()
	csm_ = physics.vector_to_csm(y_)
	csm_ = np.expand_dims(csm_,axis=0)
	nmics = physics.nmics
	csm = np.zeros((65,nmics,nmics),dtype=complex)

	ps = PowerSpectraImport(csm=csm, sample_freq=sfreq) # it is mandatory to also set the sample_freq attribute!

	freqs = ps.fftfreq()
	fftidx = freqs.searchsorted(freq)
	csm[fftidx] = csm_
	ps.csm = csm #trigger digest
	ps.ind_high = fftidx +1
	ps.ind_low 	= fftidx

	bb = BeamformerCleansc( freq_data=ps, steer=physics.sv, r_diag=False)
	pm = bb.synthetic(freqs[fftidx], 0)
	Lm = pm.real.astype("float32")
	return Lm

def cmf(y_,physics, max_iter=60):
	freq = physics.freq
	sfreq = 10*freq

	y_ = physics.unstack_complex_vector(y_).numpy()
	csm_ = physics.vector_to_csm(y_)
	csm_ = np.expand_dims(csm_,axis=0)
	nmics = physics.nmics
	csm = np.zeros((65,nmics,nmics),dtype=complex)

	ps = PowerSpectraImport(csm=csm, sample_freq=sfreq) # it is mandatory to also set the sample_freq attribute!

	freqs = ps.fftfreq()
	fftidx = freqs.searchsorted(freq)
	csm[fftidx] = csm_
	ps.csm = csm #trigger digest
	ps.ind_high = fftidx +1
	ps.ind_low 	= fftidx

	bb = BeamformerCMF( freq_data=ps, steer=physics.sv, method = 'LassoLarsBIC', max_iter=max_iter)
	pm = bb.synthetic(freqs[fftidx], 0)
	Lm = pm.real.astype("float32")
	return Lm
