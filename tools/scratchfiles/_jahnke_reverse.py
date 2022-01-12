#%%
from numpy import triu_indices, diag_indices, zeros, sqrt, all

def unstack_complex_vector(y):
	N = y.shape[0]
	assert N%2 == 0
	N = N//2
	
	y_ = y[:N] + 1j*y[-N:]
	return y_

def transform_y_to_csm(y):
	N = y.shape[0]
	N = -.5 + sqrt(.25 + N*2)
	assert N%1 == 0
	N = int(N)

	csm = zeros((N,N),dtype=complex)
	i = triu_indices(N)
	csm[i] = y
	csm_ = csm + csm.T.conj()
	csm_[diag_indices(N)] /= 2
	assert all(csm_ == csm_.T.conj())
	return csm_

def reshape_sourcemap(x):
	N = x.shape[0]
	gridsize = N
	gridlen	 = int(gridsize**0.5)

	x = x[:gridsize].reshape(gridlen,gridlen)
	return x

def transform_tensor_to_sourcemap(x):
	x = x.numpy()[0]
	x = unstack_complex_vector(x)
	x = x.real
	x = reshape_sourcemap(x)
	return x

def transform_tensor_to_vector(x):
	x = x.numpy()[0]
	x = unstack_complex_vector(x)
	return x
# %%