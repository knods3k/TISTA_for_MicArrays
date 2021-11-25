#%%
from os.path import normpath,join,pardir,dirname
from re import search
from tensorflow.python.keras.saving.saved_model.load import load
from tools.training.models import TISTA
from tools.training.bernoulli_gaussian import get_bg_batch
from tools.jahnke import create_sensing_matrix
from tensorflow.keras.models import load_model

# PATHS

HE=4
SNR=20

PATH = dirname(__file__)
MICPATH = normpath(join(PATH,pardir,"data","_tub_vogel16_ap1.xml"))
MODELDIR = "models T=20"
APATH = normpath(join(PATH, pardir,f"{MODELDIR}",f"He={HE}_SNR={SNR}",f"A_{HE}.npy"))
MODELPATH = normpath(join(PATH,pardir,pardir,f"{MODELDIR}",f"He={HE}_SNR={SNR}"))

def retrieve_snr(modelpath):
    """
    Assuming modelpath contains "SNR=NUM"
    this returns NUM
    """
    match = search("SNR=(\d+)",modelpath)
    if match:
        return match.group(1)

def retrieve_he(modelpath):
    """
    Assuming modelpath contains "HE=NUM"
    this return NUM
    """
    match = search("He=(\d+)",modelpath)
    if match:
        return match.group(1)

def retrieve_T(model):
    """
    Assuming two trainable weights per layer,
    this returns the number of layers T
    """
    return len(model.trainable_weights)

def convert_model(modelpath):
    HE = retrieve_he(modelpath)
    freq = HE*343
    A = create_sensing_matrix(freq)
    path_in     = normpath(modelpath)
    path_out    = path_in + "_converted"
    model_in    = load_model(modelpath)
    T = retrieve_T(model_in)
    model_out   = TISTA(A,T=T)
    model_out.set_weights(model_in.get_weights())
    model_out.save_weights(path_out)
    pass

convert_model(MODELPATH)




# %%
