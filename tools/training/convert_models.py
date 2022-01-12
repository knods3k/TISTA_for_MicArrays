#%%
from os.path import normpath,join,pardir,dirname
from os import listdir
from pathlib import Path
from re import search
from tensorflow.python.keras.saving.saved_model.load import load
from tools.training.models import TISTA
from tools.training.data import get_bg_batch
from tools.jahnke import create_sensing_matrix
from tensorflow.keras.models import load_model
from numpy import save

# PATHS

HE=4
SNR=40

PATH = dirname(__file__)
MICPATH = normpath(join(PATH,pardir,"data","_tub_vogel16_ap1.xml"))
MODELDIR = normpath("models T=[3,30]")
MODELNAME = normpath(f"He={HE}_SNR={SNR}")
APATH = normpath(join(PATH, pardir,f"{MODELDIR}",f"He={HE}_SNR={SNR}",f"A_{HE}.npy"))
MODELPATH = normpath(join(PATH,pardir,pardir,MODELDIR,MODELNAME))
SAVEPATH = normpath(join(MODELDIR,"converted",MODELNAME,MODELNAME))

def retrieve_snr(modelpath):
    """
    Assuming modelpath contains "SNR=NUM"
    this returns NUM
    """
    match = search("SNR=(\d+)",modelpath)
    if match:
        snr_str = match.group(1)
        return int(snr_str)

def retrieve_he(modelpath):
    """
    Assuming modelpath contains "HE=NUM"
    this returns NUM
    """
    match = search("He=(\d+)",modelpath)
    if match:
        he_str = match.group(1)
        return int(he_str)

def retrieve_T(modelpath):
    """
    Assuming modelpath contains "T=NUM"
    this returns NUM
    """
    match = search("T=(\d+)",modelpath)
    if match:
        he_str = match.group(1)
        return int(he_str)

def convert_model(path_in,path_out,modelname):
    path_out    = normpath(path_out)
    Path(path_out).mkdir(parents=True,exist_ok=True)

    path_in     = normpath(path_in)
    model_in    = load_model(modelpath)
    
    HE = retrieve_he(path_in)
    freq = HE*343
    A = create_sensing_matrix(freq)
    T = retrieve_T(model_in)

    model_out   = TISTA(A,T=T)

    model_out.set_weights(model_in.get_weights())
    model_out.save_weights(join(path_out,f"{modelname}"))
    save(join(path_out,"A.npy"),A)
    print(f"Converted model at {path_out}")
    pass

if __name__ == "__main__":

    MODELLIST = listdir(MODELDIR)
    try:
        MODELLIST.remove("converted")
    except:
        pass



    for m in MODELLIST:
        modelpath = normpath(join(PATH,pardir,pardir,MODELDIR,m))
        savepath = normpath(join(PATH,pardir,pardir,MODELDIR+"_converted",m))
        convert_model(modelpath,savepath,m)



# %%
