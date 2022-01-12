#%%
import numba
numba.set_num_threads(1)
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)

from time import time


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rc("image",cmap="hot_r")
plt.rc("font",size=14)

from tools.environment import HE, sv

from tools.model import A, model 
from tools.training.data import get_bg_batch
from tools.results.compare_cleansc import tista, cleansc
from tools.scratchfiles._jahnke_reverse import transform_tensor_to_sourcemap, transform_tensor_to_vector, transform_y_to_csm
from tools.scratchfiles._spectra_import import PowerSpectraImport

from acoular import config, BeamformerCleansc
config.global_caching="none" # disable caching!


# set up the parameters
FREQ = HE*343
SFREQ = 10*FREQ
duration = 1
nsamples = duration*SFREQ


data = get_bg_batch(A,1)
data_iter = iter(data) 

def tista(y_):
    start = time()
    x_pred = model(y_)
    x_pred = transform_tensor_to_sourcemap(x_pred)
    end = time()
    elapsed = end - start
    #L_p(x_pred)
    return (x_pred.real, elapsed)


def cleansc(y_,sfreq=SFREQ,freq=FREQ):
    y_ = transform_tensor_to_vector(y_)
    csm_ = transform_y_to_csm(y_)
    csm_ = np.expand_dims(csm_,axis=0)
    csm = np.zeros((65,16,16),dtype=complex)

    ps = PowerSpectraImport(csm=csm, sample_freq=sfreq) # it is mandatory to also set the sample_freq attribute!

    freqs = ps.fftfreq()
    fftidx = freqs.searchsorted(freq)
    csm[fftidx] = csm_
    ps.csm = csm #trigger digest

    ps.ind_high = fftidx +1
    ps.ind_low 	= fftidx

    #ps = PowerSpectraImport(csm=csm, sample_freq=sfreq) # it is mandatory to also set the sample_freq attribute!

    start = time()
    bb = BeamformerCleansc( freq_data=ps, steer=sv, r_diag=False)
    #print(bb.digest)
    pm = bb.synthetic( freqs[fftidx], 0 )
    end = time()
    elapsed = end - start
    #Lm = L_p( pm ).T
    Lm = pm.real.astype("float32")
    return (Lm, elapsed)


num_runs = 100000



if __name__ == "__main__":

    duration_tista      = np.zeros(num_runs)
    duration_cleansc    = np.zeros(num_runs)

    for i in range(num_runs):
        y,x = next(data_iter)
        duration_tista[i]   = tista(y)[1]
        duration_cleansc[i] = cleansc(y)[1]

#%%
    fig, (ax,ax1) = plt.subplots(2,1, sharey=True,sharex=True,figsize=(16,9),dpi=100)
    
    bins=np.histogram(np.hstack((duration_tista,duration_cleansc)), bins=200)[1]
    ax.hist(duration_tista, bins=bins, log=True)
    ax.set_ylabel("Statistical Frequency")
    ax.set_title("TISTA")

    ax1.hist(duration_cleansc, bins=bins, log=True)
    ax1.set_title("CLEAN-SC")
    ax1.set_ylabel("Statistical Frequency")
    ax1.set_xlabel("Computation Time [s]")

    fig.show()

    durations = pd.DataFrame({"TISTA": duration_tista, "CLEAN-SC": duration_cleansc})
    print(durations.describe())

# %%
    fig.savefig("data/plots/time_hist.pdf")
# %%
