#%%
from os.path import normpath, join
from acoular import config, L_p, BeamformerCleansc, BeamformerCMF
from tools.scratchfiles._spectra_import import PowerSpectraImport
from tools.scratchfiles._jahnke_reverse import *
import numpy as np
from tensorflow import einsum
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

from tools.environment import HE, SNR, sv
# from tools.model import model, A, data
from tools.physical import physics
from tools.training.data import DataGenerator
from tools.pyplot_setup import params

from scipy.signal import convolve2d

config.global_caching="none" # disable caching
plt.rcParams.update(params)
#%%
HE = 16
T = 40
BASEPATH = normpath(join("models","Clean"))
MODELNAME = f"He={HE}_T={T}"
PATH = normpath(join(f"{BASEPATH}",f"{MODELNAME}"))
model = load_model(PATH)


random_matrix_path = normpath(join("data", "random_matrices", f"{HE}"))
As = random_matrix_path

# set up the parameters
NOISY = False
NSOURCES = 10
FREQ = HE*343
SFREQ = 10*FREQ
duration = 1
nsamples = duration*SFREQ

A = physics.A
pnz = NSOURCES / A.shape[1]
generator = DataGenerator(As, batchsize=1, pnz=pnz, noise=NOISY)
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

def tista(y_):
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


def cmf(y_,sfreq=SFREQ,freq=FREQ):
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


	bb = BeamformerCMF( freq_data=ps, steer=physics.sv)
	#print(bb.digest)
	pm = bb.synthetic(freqs[fftidx], 0)
	#Lm = L_p( pm ).T
	Lm = pm.real.astype("float32")
	return Lm


if __name__ == "__main__" and NOISY == True:
	x_true = physics.vector_to_sourcemap(x)
	x_true = L_p(x_true)

	x_cleansc_noise		= L_p(cmf(y_simu))
	x_cleansc 			= L_p(cmf(y_calc))
	x_tista_noise 		= L_p(tista(y_simu))
	x_tista 			= L_p(tista(y_calc))



	fig, axs = plt.subplots(3,2,sharex=True,sharey=True,figsize=(9,16),dpi=100)
	im = axs[0][0].imshow(x_true,origin="lower",vmax=95,vmin=75,extent=[-.5,.5,-.5,.5])
	axs[0][0].set_ylabel("True")
	axs[0][0].set_title(f"CLEAN-SC")# (t = {cleansc_t} s)")
	axs[0][1].imshow(x_true,origin="lower",vmax=95,vmin=75,extent=[-.5,.5,-.5,.5])
	axs[0][1].set_title(f"TISTA")# (t = {tista_t} s)")
	axs[0][1].set_ylabel("y")
	axs[0][1].yaxis.tick_right()
	axs[0][1].yaxis.set_label_position("right")
	axs[1][0].imshow(x_cleansc,origin="lower",vmax=95,vmin=75,extent=[-.5,.5,-.5,.5])
	axs[1][0].set_ylabel("Predicted")
	axs[1][1].imshow(x_tista,origin="lower",vmax=95,vmin=75,extent=[-.5,.5,-.5,.5])
	axs[1][1].set_ylabel("y")
	axs[1][1].yaxis.tick_right()
	axs[1][1].yaxis.set_label_position("right")
	axs[2][0].imshow(x_cleansc_noise,origin="lower",vmax=95,vmin=75,extent=[-.5,.5,-.5,.5])
	axs[2][0].set_ylabel("Predicted (with Noise)")
	axs[2][0].set_xlabel("x")
	axs[2][1].imshow(x_tista_noise,origin="lower",vmax=95,vmin=75,extent=[-.5,.5,-.5,.5])
	axs[2][1].set_xlabel("x")
	axs[2][1].set_ylabel("y")
	axs[2][1].yaxis.tick_right()
	axs[2][1].yaxis.set_label_position("right")

	cbar = fig.colorbar(im, ax=axs.ravel().tolist(),orientation="horizontal")
	cbar.ax.set_title("Sound Pressure Level [dB]")
	#fig.suptitle(f"He={HE} | SNR={SNR} | NMICS=16 | GRID=26x26 | T=30")
			#%%
if __name__ == "__main__" and NOISY == False:
	y,x = next(iter(generator.get_batch()))
	#x[x != reduce_max(x)] = 0
	y_simu = y
	y_calc = einsum("ij,kj->ki",A,x)

	m = 40
	x_true = transform_tensor_to_sourcemap(x).real
	x_true = (L_p(magnify(x_true, scale=m)))
	x_cleansc 	= (L_p(magnify(cleansc(y_calc), scale=m)))
	x_tista		= (L_p(magnify(tista(y_calc), scale=m)))


	SNR = "Inf"
	fig, axs = plt.subplots(1,3,sharex=True,sharey=True,figsize=(15,5),dpi=100)
	im = axs[0].imshow((x_true),origin="lower",vmax=100,vmin=80,extent=[-.5,.5,-.5,.5])
	axs[0].set_title("True")
	axs[0].set_xlabel("x")
	axs[0].set_ylabel("y")
	axs[1].imshow((x_tista),origin="lower",vmax=95,vmin=75,extent=[-.5,.5,-.5,.5])
	axs[1].set_title(f"TISTA")
	axs[1].set_xlabel("x")
	axs[2].imshow((x_cleansc),origin="lower",vmax=95,vmin=75,extent=[-.5,.5,-.5,.5])
	axs[2].set_title(f"CLEAN-SC")
	axs[2].set_xlabel("x")

	cbar = fig.colorbar(im, ax=axs.ravel().tolist())
	cbar.ax.yaxis.set_label_position('left')
	cbar.ax.set_ylabel("Sound Pressure Level [dB]")
	#fig.suptitle(f"He={HE} | SNR={SNR} | NMICS=16 | GRID=26x26 | T=30")
#%%
	#fig.savefig(f"data/plots/sourcemaps_he{HE}.pdf")

#%%
	fig.savefig(f"data/plots/sourcemaps_he{HE}_nonoise.pdf")

#%%



	plt.figure()
	plt.imshow(L_p(x_true),origin="lower",vmax=95,vmin=75)
	plt.colorbar()
	plt.figure()
	plt.imshow(L_p(sourcemaps[1].real),origin="lower",vmax=95,vmin=75)
	plt.title("Calculated")
	plt.colorbar()
	plt.figure()
	plt.imshow(L_p(sourcemaps[0].real),origin="lower",vmax=95,vmin=75)
	plt.title("Simulated")
	plt.colorbar()


#%%

	for y_ in [y_simu,y_calc]:
		y_ = y_.numpy()
		y_ = y_[0]
		y_ = unstack_complex_vector(y_)
		csm_ = transform_y_to_csm(y_)
		csm_ = expand_dims(csm_,axis=0)
		csm = zeros((65,16,16),dtype=complex)


		ps = PowerSpectraImport(csm=csm, sample_freq=sfreq) # it is mandatory to also set the sample_freq attribute!

		freqs = ps.fftfreq()
		fftidx = freqs.searchsorted(freq)
		csm[fftidx] = csm_
		
		# analyze the data and generate map
		# ts = TimeSamples( name=h5savefile )
		#ps = PowerSpectra( time_data=ts, block_size=128, window='Hanning' )


		bb = BeamformerCleansc( freq_data=ps, steer=sv, r_diag=True)
		pm = bb.synthetic( freqs[fftidx], 0 )
		Lm = L_p( pm ).T
		sourcemaps.append(Lm)


	plt.figure()
	plt.imshow(L_p(x_smap),origin="lower",vmax=95,vmin=75)
	plt.colorbar()
	plt.figure()
	plt.imshow(sourcemaps[1],origin="lower",vmax=95,vmin=75)
	plt.title("Calculated")
	plt.colorbar()
	plt.figure()
	plt.imshow(sourcemaps[3],origin="lower",vmax=95,vmin=75)
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
