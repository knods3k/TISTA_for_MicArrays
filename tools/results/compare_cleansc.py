#%%
import tools
#%%
from os.path import normpath, join
from acoular import config, L_p, BeamformerCleansc, BeamformerCMF
from tools.scratchfiles._spectra_import import PowerSpectraImport
from tools.scratchfiles._jahnke_reverse import *
import numpy as np
from tensorflow import einsum
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

from tools.environment import INCREMENT, NMICS
# from tools.model import model, A, data
from tools.physical import PhyiscalModel
from tools.training.data import DataGenerator
from tools.pyplot_setup import params

from scipy.signal import convolve2d

config.global_caching="none" # disable caching
plt.rcParams.update(params)
imshow_kwargs = {"origin":"lower", "extent":[-.5,.5,-.5,.5],"vmax":95,"vmin":75, "interpolation":None}
#%%
HE = 4
T = 60
BASEPATH = normpath(join("models","convergence",f"He={HE}"))
MODELNAME = f"He={HE}_T={T}"
PATH = normpath(join(f"{BASEPATH}",f"{MODELNAME}"))
model = load_model(PATH)


random_matrix_path = normpath(join("data", "random_matrices", f"{HE}"))
As = random_matrix_path

# set up the parameters
NOISY = True
NSOURCES = 10
FREQ = HE*343
SFREQ = 10*FREQ
duration = 1
nsamples = duration*SFREQ

physics = PhyiscalModel(HE, INCREMENT, NMICS)


A = physics.A
# pnz = NSOURCES / A.shape[1]
generator = DataGenerator(As, batchsize=1, nsources=NSOURCES, noise=NOISY)
y,x = next(iter(generator.get_batch()))
#x[x != reduce_max(x)] = 0
y_simu = y
y_calc = einsum("ij,kj->ki",A,x)


#%%

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

def tista(y_, model = model):
	x_pred = model(y_)
	x_pred = physics.vector_to_sourcemap(x_pred)
	#L_p(x_pred)
	return x_pred


def cleansc(y_,sfreq=SFREQ,freq=FREQ):
	# y_ = transform_tensor_to_vector(y_)
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

	#ps = PowerSpectraImport(csm=csm, sample_freq=sfreq) # it is mandatory to also set the sample_freq attribute!


	bb = BeamformerCleansc( freq_data=ps, steer=physics.sv, r_diag=False)
	#print(bb.digest)
	pm = bb.synthetic(freqs[fftidx], 0)
	#Lm = L_p( pm ).T
	Lm = pm.real.astype("float32")
	return Lm


def cmf(y_,sfreq=SFREQ,freq=FREQ, max_iter=60):
	# y_ = transform_tensor_to_vector(y_)
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

	#ps = PowerSpectraImport(csm=csm, sample_freq=sfreq) # it is mandatory to also set the sample_freq attribute!


	bb = BeamformerCMF( freq_data=ps, steer=physics.sv, method = 'LassoLarsBIC', max_iter=max_iter)
	#print(bb.digest)
	pm = bb.synthetic(freqs[fftidx], 0)
	#Lm = L_p( pm ).T
	Lm = pm.real.astype("float32")
	return Lm


#%%
if __name__ == "__main__" and NOISY == True:
	x_true 				= physics.vector_to_sourcemap(x)
	x_true 				= (L_p(bin(x_true)))
	x_cleansc 			= (L_p(bin(cleansc(y_calc))))
	x_cleansc_noise 	= (L_p(bin(cleansc(y_simu))))
	x_cmf	 			= (L_p(bin(cmf(y_calc))))
	x_cmf_noise	 		= (L_p(bin(cmf(y_simu))))
	x_tista				= (L_p(bin(tista(y_calc))))
	x_tista_noise		= (L_p(bin(tista(y_simu))))


	ratio = 16/9
	width = 451.6875 / 72.27
	height = width / ratio
	width = width * ratio

	plt.figure(figsize=(width,width/2))
	plt.subplot(231)
	plt.imshow(x_true)
	plt.subplot(232)
	plt.subplot(233)
	plt.subplot(234)
	plt.subplot(235)
	plt.subplot(236)
	plt.tight_layout()
	plt.savefig('data/plots/conv_nsources.pdf')



#$$
	fig, axs = plt.subplots(3,2,sharex=True,sharey=True,figsize=(9,16),dpi=100)
	im = axs[0][0].imshow(x_true,**imshow_kwargs)
	axs[0][0].set_ylabel("True")
	axs[0][0].set_title(f"CMF")# (t = {cleansc_t} s)")
	axs[0][1].imshow(x_true,**imshow_kwargs)
	axs[0][1].set_title(f"TISTA")# (t = {tista_t} s)")
	axs[0][1].set_ylabel("y")
	axs[0][1].yaxis.tick_right()
	axs[0][1].yaxis.set_label_position("right")
	axs[1][0].imshow(x_cmf,**imshow_kwargs)
	axs[1][0].set_ylabel("Predicted")
	axs[1][1].imshow(x_tista,**imshow_kwargs)
	axs[1][1].set_ylabel("y")
	axs[1][1].yaxis.tick_right()
	axs[1][1].yaxis.set_label_position("right")
	axs[2][0].imshow(x_cmf_noise,**imshow_kwargs)
	axs[2][0].set_ylabel("Predicted (with Noise)")
	axs[2][0].set_xlabel("x")
	axs[2][1].imshow(x_tista_noise,**imshow_kwargs)
	axs[2][1].set_xlabel("x")
	axs[2][1].set_ylabel("y")
	axs[2][1].yaxis.tick_right()
	axs[2][1].yaxis.set_label_position("right")

	plt.tight_layout()

	cbar = fig.colorbar(im, ax=axs.ravel().tolist(),orientation="horizontal")
	cbar.ax.set_title("Sound Pressure Level [dB]")
	# fig.suptitle(f"He={HE} | NMICS={NMICS} | GRID=51x51 | T={T}")
	#%%
if __name__ == "__main__" and NOISY == False:
	y,x = next(iter(generator.get_batch()))
	#x[x != reduce_max(x)] = 0
	y_simu = y
	y_calc = einsum("ij,kj->ki",A,x)
	# y_calc = y

	m = 2
	x = transform_tensor_to_sourcemap(x).real
	x_true = (L_p((x)))
	x_cmf	 	= (L_p((cmf(y_calc))))
	x_tista		= (L_p((tista(y_calc))))
	
	x_true_b 	= (L_p(bin(x)))
	x_cmf_b	 	= (L_p(bin(cmf(y_calc))))
	x_tista_b	= (L_p(bin(tista(y_calc))))


	SNR = "Inf"
	fig, axs = plt.subplots(1,3,sharex=True,sharey=True,figsize=(15,5),dpi=100)
	im = axs[0].imshow((x_true_b),**imshow_kwargs)
	axs[0].set_title("True")
	axs[0].set_xlabel("x")
	axs[0].set_ylabel("y")
	axs[1].imshow((x_tista_b),**imshow_kwargs)
	axs[1].set_title(f"TISTA")
	axs[1].set_xlabel("x")
	axs[2].imshow((x_cmf_b),**imshow_kwargs)
	axs[2].set_title(f"CMF")
	axs[2].set_xlabel("x")

	cbar = fig.colorbar(im, ax=axs.ravel().tolist())
	cbar.ax.yaxis.set_label_position('left')
	cbar.ax.set_ylabel("Sound Pressure Level [dB]")
	fig.suptitle(f"He={HE} | NMICS={NMICS} | GRID=51x51 | T={T}")
#%%
	#fig.savefig(f"data/plots/sourcemaps_he{HE}.pdf")


#%%