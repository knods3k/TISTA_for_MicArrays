#%%
from os.path import dirname, join, pardir, normpath
from tools.training.data import get_bg_batch
from tensorflow.keras.models import load_model
from jahnke import create_sensing_matrix
from numpy import load
from environment import HE, SNR, FREQ


# PATHS

PATH = dirname(__file__)
MODELDIR = "models T=30"
MODELPATH = normpath(join(PATH,pardir,f"{MODELDIR}",f"He={HE}_SNR={SNR}"))
APATH = normpath(join(MODELPATH,f"A_{HE}.npy"))




# MODEL

try:
    A = load(APATH)
    model = load_model(MODELPATH)
    data = get_bg_batch(A,1,SNR=SNR)
except:
    #print(e.message)
    A = create_sensing_matrix(FREQ)
    data = get_bg_batch(A,1,SNR=SNR)
    print("Model doesn't exist")


# %%
