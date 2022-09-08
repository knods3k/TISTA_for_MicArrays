#%%

#%%
from os.path import dirname, join, pardir, normpath
from acoular import MicGeom, RectGrid, SteeringVector
from numpy import argmin, sum, abs, load, Infinity

# SETUP

SNR=40
HE=16
FREQ = HE*343
NMICS = 64
INCREMENT = 0.01




# PATHS

PATH = dirname(__file__)
MICPATH = normpath(join(PATH,pardir,"data",f"_tub_vogel{NMICS}_ap1.xml"))
MODELDIR = "models"
APATH = normpath(join(PATH, pardir,f"{MODELDIR}",f"He={HE}_SNR={SNR}",f"A_{HE}.npy"))
MODELPATH = normpath(join(PATH,pardir,f"{MODELDIR}",f"He={HE}_SNR={SNR}"))




# MESAUREMENT ENVIRONMENT
#%%

mg = MicGeom(from_file=MICPATH)
rg = RectGrid(x_max=0.5,x_min=-0.5,y_max=0.5,y_min=-0.5,z=0.5,increment=INCREMENT)
sv	= SteeringVector(grid=rg, mics=mg)
ref_mic_idx = argmin(sum(abs(mg.mpos),axis=1))
ref_mic_pos = mg.mpos.T[ref_mic_idx]
sv._set_ref(ref_mic_pos)				

# %%
