#%%
from acoular import config, L_p, BeamformerCleansc

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from matplotlib import pyplot as plt

from os.path import dirname,pardir,normpath,join

from tools.environment import sv, rg 
from tools.scratchfiles._spectra_import import PowerSpectraImport
from tools.scratchfiles._jahnke_reverse import *
from tools.scratchfiles._evaluate import PlanarSourceMapEvaluator
from tools.training.data import get_bg_batch
from tools.jahnke import create_sensing_matrix

config.global_caching="none"
plt.rc("image",cmap="hot_r")
plt.rc("font",size=14)

df = pd.DataFrame()



def cleansc(y_,sfreq,freq):
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


    bb = BeamformerCleansc( freq_data=ps, steer=sv, r_diag=False)
    #print(bb.digest)
    pm = bb.synthetic( freqs[fftidx], 0 )
    #Lm = L_p( pm ).T
    Lm = pm.real.astype("float32")
    return Lm


for HE in [4,8,16]:
    # SET PARAMETERS
    FREQ = HE*343
    SNR = 10
    SFREQ = 10*FREQ



    # LOAD MODEL
    PATH = dirname(__file__)
    MODELDIR = "models T=30"
    MODELPATH = normpath(join(PATH,pardir,pardir,f"{MODELDIR}",f"He={HE}_SNR={SNR}"))
    model = load_model(MODELPATH)


    A = create_sensing_matrix(FREQ)
    data = get_bg_batch(A,1,SNR=SNR)
    y,x = next(iter(data))

    # CREATE EXAMPLE WITH ARBITRARY LOCATION/SOURCESTRENGTH

    S1 = .66
    S2 = 1.01
    S3 = 1.9

    S1_idx = int(S1*100)
    S2_idx = int(S2*200)
    S3_idx = int(S3*300)

    x= np.zeros(x.numpy().shape[1])
    x[S1_idx] = S1
    x[S2_idx] = S2
    x[S3_idx] = S3

    y = np.matmul(A,x)
    noise_var = (10**(-SNR/10)) * np.mean(np.abs(y))
    y = y + np.random.normal(0,noise_var,y.shape)

    x  = np.expand_dims(x,axis=0)
    x = tf.convert_to_tensor(x)
    y = tf.einsum("ij,kj->ki",A,x)

    smap = transform_tensor_to_sourcemap(x)


    # SET UP EVALUATOR

    i,j,k = rg.pos()
    ps1_loc = [i[S1_idx],j[S1_idx],k[S1_idx]]
    ps2_loc = [i[S2_idx],j[S2_idx],k[S2_idx]]
    ps3_loc = [i[S3_idx],j[S3_idx],k[S3_idx]]

    ps1_p2 = S1
    ps2_p2 = S2
    ps3_p2 = S3

    target_loc = np.array([ps1_loc,ps2_loc,ps3_loc])
    target_p2 = np.array([[ps1_p2,ps2_p2,ps3_p2]])

    eval = PlanarSourceMapEvaluator()
    eval.grid = rg
    eval.target_loc = target_loc
    eval.target_p2 = target_p2


    # EVALUATE TISTA
    x_tista = model(y)
    smap_tista = transform_tensor_to_sourcemap(x_tista)
    eval.sourcemap = np.array([smap_tista])

    overall_tista = eval.get_overall_level_error()
    specific_tista = eval.get_specific_level_error()
    inverse_tista = eval.get_inverse_level_error()

    print(overall_tista,specific_tista,inverse_tista)


    # EVALUATE CLEAN_SC
    x_clean = cleansc(y,freq=FREQ,sfreq=SFREQ)
    smap_clean = x_clean
    eval.sourcemap = np.array([smap_clean])

    overall_clean = eval.get_overall_level_error()
    specific_clean = eval.get_specific_level_error()
    inverse_clean = eval.get_inverse_level_error()

    print(overall_clean,specific_clean,inverse_clean)



    # STATISTICS
    errors_tista = {"overall": overall_tista, "specific": specific_tista, "inverse": inverse_tista}
    errors_clean = {"overall": overall_clean, "specific": specific_clean, "inverse": inverse_clean}


    df = df.append(pd.DataFrame({f"tista_{HE}": errors_tista, f"clean_{HE}": errors_clean}).T)
df = df.applymap(np.round,decimals=3)
df = df.applymap(lambda x: x[0])

print(df.to_latex())



#%%

# PLOT TEST CASE

fig, ax = plt.subplots()
im = ax.imshow(L_p(smap.T),origin="lower",vmax=95,vmin=75,extent=[-.5,.5,-.5,.5])
r = eval.r
c1 = plt.Circle((ps1_loc[0],ps1_loc[1]),r,color="k",fill=False)
c2 = plt.Circle((ps2_loc[0],ps2_loc[1]),r,color="k",fill=False)
c3 = plt.Circle((ps3_loc[0],ps3_loc[1]),r,color="k",fill=False)
ax.add_patch(c1)
ax.add_patch(c2)
ax.add_patch(c3)
ax.set_xlabel("x")
ax.set_ylabel("y")
cbar = fig.colorbar(im)
cbar.set_label("Sound Pressure Level [dB]")
cbar.ax.yaxis.set_label_position("left")

# %%
fig.savefig("data/plots/test_case.pdf")
# %%
