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

# %%