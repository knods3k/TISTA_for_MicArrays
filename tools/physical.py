#%%
from tkinter import N
import numpy as np
import tensorflow as tf
from acoular import SteeringVector, MicGeom, RectGrid
from os.path import dirname, join, pardir, normpath

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

# this file is intended to help with implementing the formulation for linear
# optimization for microphone array data such that y = A * x found in Alexander
# Jahnke's masters thesis
#%%

HE = 16
INCREMENT = 0.01
NMICS = 64

class PhyiscalModel():
	def __init__(self, he=None, increment=0.04, nmics=16):
		
		self.he = he
		self.freq = he * 343
		self.increment = increment
		self.nmics = nmics
		self.mg = MicGeom(from_file=self.mic_path)

		self.M, self.N = self.A.shape

	@property
	def	mic_path(self):
		return normpath(join("data",f"_tub_vogel{self.nmics}_ap1.xml"))
	
	# @property
	# def mg(self):
	# 	return MicGeom(from_file=self.mic_path)
	
	@property
	def rg(self):
		return RectGrid(x_max=0.5,x_min=-0.5,y_max=0.5,y_min=-0.5,z=0.5,\
			increment=self.increment)
	
	@property
	def sv(self):
		sv = SteeringVector(grid=self.rg, mics=self.mg)
		ref_mic_idx = np.argmin(np.sum(np.abs(self.mg.mpos),axis=1))
		ref_mic_pos = self.mg.mpos.T[ref_mic_idx]
		sv.ref = ref_mic_pos
		return sv

	@property
	def indices(self):
		return np.triu_indices(self.mg.num_mics)

	@property
	def A(self):
		tv = self.sv.transfer(self.freq).astype(np.complex64)
		tm	= np.einsum("ij,il -> ijl",tv,np.conj(tv))
		A = tm[:,self.indices[0],self.indices[1]].T
		A = self.stack_complex_matrix(A)
		A = tf.convert_to_tensor(A, dtype=tf.float32)
		return A

	@staticmethod
	def stack_complex_matrix(A):
		A_real	= tf.concat((tf.math.real(A), tf.math.imag(-A)), axis=1)
		A_imag	= tf.concat((tf.math.imag(A), tf.math.real(A)), axis=1)
		A_hat	= tf.concat((A_real, A_imag), axis=0)
		return A_hat

	@staticmethod
	def stack_complex_vector(v):
		v_hat = tf.concat((tf.math.real(v), tf.math.imag(v)), axis=0)
		return v_hat

	@staticmethod
	def unstack_complex_vector(v):
		L = v.shape[1]
		assert L%2 == 0
		L = L//2

		re = v[0][:L]
		im = v[0][-L:]		
		v_hat = tf.complex(re, im)
		return v_hat

	@staticmethod
	def csm_to_vector(csm, i):
		return csm[i]


	def vector_to_csm(self, v):
		M = self.nmics
		csm = np.zeros((M, M),dtype=complex)
		i = np.triu_indices(M)
		csm[i] = v
		csm_ = csm + np.conj(csm)
		i = np.diag_indices(M)
		csm_[i] /= 2
		assert np.all(csm_ == np.conj(csm_))
		return csm_

	@staticmethod
	def reshape_sourcemap(x):
		N = x.shape[0]
		gridlen	 = int(np.floor(N**0.5))
		assert gridlen**2 == N

		x = x.reshape(gridlen,gridlen)
		return x

	@staticmethod
	def sourcemap_to_vector(x):
		x = x.reshape(-1)
		N = x.shape[0]
		paddings = tf.constant([[0,N]])
		x = tf.pad(x,paddings)
		x = x.reshape(1,-1)
		return x

	# @staticmethod
	def vector_to_sourcemap(self,x):
		x = self.unstack_complex_vector(x)
		x = tf.math.real(x)
		x = self.reshape_sourcemap(x)
		return x



physics = PhyiscalModel(HE, INCREMENT, NMICS)

#%%


# def find_indices(mg):
# 	"""return upper triangle indices for mg

# 	mg should be acoular MicGeom object
	
# 	to be used with jahnke.transform_tm__to_A(), jahnke.transform_csm_to_y()
# 	"""
# 	return np.triu_indices(mg.num_mics)

# def transform_tm_to_A (tm,i):
# 	"""
# 	return 

# 	tm should be acoular transfer matrix
# 	i should be output of jahnke.find_indices()

# 	if used accordingly this will return matrix A of jahnke formulation
# 	such that y = A*x
# 	"""
# 	A = tm[:,i[0],i[1]].T
# 	return A

# def transform_rg_mg_to_A (rg,mg,freq,i):
# 	"""
# 	return sensing matrix A

# 	rg should be acoular RectGrid object
# 	mg should be acoular MicGeom object
# 	freq should be frequency for which to calculate A
# 	i should be output of jahnke.find_indices()

# 	if used accordingly this will return matrix A of jahnke formulation
# 	such that y = A*x
# 		"""
# 	sv	= SteeringVector(grid=rg, mics=mg)			# steering vector
# 	ref_mic_idx = np.argmin(np.sum(np.abs(mg.mpos),axis=1))
# 	ref_mic_pos = mg.mpos.T[ref_mic_idx]
# 	sv._set_ref(ref_mic_pos)						# set reference position to microphone closest ot origin
# 	tv	= sv.transfer(freq)							# transfer matrix
# 	tm	= np.einsum("ij,il -> ijl",tv,tv.conj())	# transfer matrix remodelling
# 	A = tm[:,i[0],i[1]].T
# 	return A

# def create_sensing_matrix(freq):

#     i = find_indices(mg)
#     A = stack_complex_matrix(transform_sv_to_A(sv,freq,i)).astype(np.float32)
#     return A

# def transform_sv_to_A (sv,freq,i):
# 	"""
# 	return sensing matrix A

# 	sv should be acoular SteeringVector object
# 	freq should be frequency for which to calculate A
# 	i should be output of jahnke.find_indices()

# 	if used accordingly this will return matrix A of jahnke formulation
# 	such that y = A*x
# 		"""
# 	tv	= sv.transfer(freq)							# transfer matrix
# 	tm	= np.einsum("ij,il -> ijl",tv,tv.conj())	# transfer matrix remodelling
# 	A = tm[:,i[0],i[1]].T
# 	return A

# def transform_csm_to_y (csm,i):
# 	"""
# 	return 1d array with elements i of csm

# 	csm should be acoular cross spectral matrix
# 	i should be output of jahnke.find_indices()

# 	if used accordingly this will return vector y of jahnke formulation
# 	such that y = A*x
# 	"""
# 	y = csm[i]
# 	return y

# def transform_ps_to_y (ps,fidx,i):
# 	"""
# 	return 1d array with elements i of csm at fidx

# 	csm should be acoular cross spectral matrix
# 	i should be output of jahnke.find_indices()
# 	fidx should be index of desired frequency

# 	if used accordingly this will return vector y of jahnke formulation
# 	such that y = A*x
# 	"""
# 	y = ps.csm[fidx][i]
# 	return y

# def transform_pn_list_to_x (pn_list,pa2_list,rg):
# 	"""
# 	return sparse 1d array with pa2 elements at pn locations on rg

# 	pn_list should be list of acoular pointsources
# 	pa2_list should be list of respective soundpressure squares
# 	rg should be acoular rectangular grid

# 	if used accordingly this will return vector x of jahnke formulation
# 	such that y = A*x
# 	"""
# 	x = np.zeros(rg.shape)
# 	for idx,pn in enumerate(pn_list):
# 		x[rg.index(pn.loc[0],pn.loc[1])] = pa2_list[idx]
# 	x = x.flatten()
# 	return x

# def stack_complex_matrix(A):
# 	"""
# 	returns 2d array of dtype float with real and imaginary parts of 2d array
# 	of dtype complex

# 	A should be output of jahnke.transform_rg_mg_to_A()

# 	if used accordingly this will return fully real matrix to be used with
# 	backpropagation algorithms such that y_hat = A_hat * x_hat
# 	"""
# 	A_real	= np.append(A.real,-A.imag,axis=1)
# 	A_imag	= np.append(A.imag,A.real,axis=1)

# 	A_hat	= np.append(A_real,A_imag,axis=0)
# 	return A_hat

# def stack_complex_vector(v):
# 	"""
# 	returns 1d array of dtype float with real and imaginary parts of 1d array
# 	of dtype complex

# 	v should be output of either jahnke.transform_pn_list_to_x() or
# 	jahnke.transform_csm_to_y()

# 	if used accordingly this will return fully real vector to be used with
# 	backgropagation algorithms such that y_hat = A_hat * x_hat
# 	"""
# 	v_hat = np.append(v.real,v.imag,axis=0)
# 	return v_hat


# def stack_complex_tensor(v):
# 	v_hat = tf.concat((tf.math.real(v), tf.math.imag(v)), axis=0)
# 	return v_hat


# # REVERSE


# def unstack_complex_vector(y):
# 	N = y.shape[0]

# 	N = N//2

# 	re = y[:N]
# 	im = y[-N:]
# 	y_ = re + 1j*im

# 	return y_


# def unstack_complex_tensor(y):
# 	N = y.shape[1]
# 	assert N%2 == 0

# 	N = N//2

# 	re = y[0][:N]
# 	im = y[0][-N:]		
# 	y_ = tf.complex(re, im)
# 	return y_


# def transform_y_to_csm(y):
# 	N = y.shape[0]
# 	N = -.5 + np.sqrt(.25 + N*2)
# 	assert N%1 == 0
# 	N = int(N)

# 	csm = np.zeros((N,N),dtype=complex)
# 	i = np.triu_indices(N)
# 	csm[i] = y
# 	csm_ = csm + csm.T.conj()
# 	csm_[np.diag_indices(N)] /= 2
# 	assert all(csm_ == csm_.T.conj())
# 	return csm_

# def reshape_sourcemap(x):
# 	N = x.shape[0]
# 	gridsize = N
# 	gridlen	 = int(np.ceil(gridsize**0.5))

# 	x = x[:gridsize].reshape(gridlen,gridlen)
# 	return x

# def transform_tensor_to_sourcemap(x):
# 	x = x.numpy()[0]
# 	x = unstack_complex_vector(x)
# 	x = x.real
# 	x = reshape_sourcemap(x)
# 	return x

# def transform_graphtensor_to_sourcemap(x):
# 	x = unstack_complex_tensor(x)
# 	x = tf.math.real(x)
# 	x = reshape_sourcemap(x)
# 	return x

# def transform_tensor_to_vector(x):
# 	x = x.numpy()[0]
# 	x = unstack_complex_vector(x)
# 	return x
# %%
