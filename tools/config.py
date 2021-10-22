from bernoulli_gaussian import get_bg_batch
from tensorflow.keras.models import load_model
from acoular import MicGeom, RectGrid, SteeringVector
from numpy import argmin, sum, abs, load



# MESAUREMENT ENVIRONMENT

mg = MicGeom(from_file="../data/_tub_vogel16_ap1.xml")
rg = RectGrid(x_max=0.5,x_min=-0.5,y_max=0.5,y_min=-0.5,z=0.5,increment=0.04)
sv	= SteeringVector(grid=rg, mics=mg)
ref_mic_idx = argmin(sum(abs(mg.mpos),axis=1))
ref_mic_pos = mg.mpos.T[ref_mic_idx]
sv._set_ref(ref_mic_pos)				



# MODEL

SNR=20
HE=16

try:
    A = load(f"../models/He={HE}_SNR={SNR}/A_{HE}.npy")
    model = load_model(f"../models/He={HE}_SNR={SNR}")
except:
    print("Model doesn't exist")

data = get_bg_batch(A,1,SNR=SNR)
