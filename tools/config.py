#%%
from os.path import dirname, join, pardir, normpath
from tools.training.bernoulli_gaussian import get_bg_batch
from tensorflow.keras.models import load_model
from acoular import MicGeom, RectGrid, SteeringVector
from numpy import argmin, sum, abs, load
#from jahnke import create_sensing_matrix

# SETUP

SNR=20
HE=8
freq = HE*343



# PATHS

PATH = dirname(__file__)
MICPATH = normpath(join(PATH,pardir,"data","_tub_vogel16_ap1.xml"))
MODELDIR = "models"
APATH = normpath(join(PATH, pardir,f"{MODELDIR}",f"He={HE}_SNR={SNR}",f"A_{HE}.npy"))
MODELPATH = normpath(join(PATH,pardir,f"{MODELDIR}",f"He={HE}_SNR={SNR}"))




# MESAUREMENT ENVIRONMENT

mg = MicGeom(from_file=MICPATH)
rg = RectGrid(x_max=0.5,x_min=-0.5,y_max=0.5,y_min=-0.5,z=0.5,increment=0.04)
sv	= SteeringVector(grid=rg, mics=mg)
ref_mic_idx = argmin(sum(abs(mg.mpos),axis=1))
ref_mic_pos = mg.mpos.T[ref_mic_idx]
sv._set_ref(ref_mic_pos)				



# MODEL

try:
    A = load(APATH)
    model = load_model(MODELPATH)
    data = get_bg_batch(A,1,SNR=SNR)
except Exception as e:
    #print(e.message)
    #A = create_sensing_matrix(sv,mg,freq)
    #data = get_bg_batch(A,1,SNR=SNR)
    print("Model doesn't exist")


# %%
