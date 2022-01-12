#%%
from numpy import load, ndarray
from os.path import dirname, join, pardir, normpath, isdir
from os import listdir, walk
from matplotlib import pyplot as plt
from tools.training.convert_models import retrieve_snr, retrieve_he, retrieve_T
import pandas as pd


plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams["figure.dpi"] = 100
plt.rc("font",size=14)

FILEPATH = dirname(__file__)
BASEPATH = normpath(join(FILEPATH,pardir,pardir,"models T=30"))


HISTORY_PATH = normpath(\
                "C:\\Users\\kaysec\\VSCodeProjects\\KERAS_Train\\models He=16\\He=16_SNR=999_T=02\\history.npy"
                        )
HISTORY_PATH = normpath(HISTORY_PATH)
history = load(HISTORY_PATH,allow_pickle=True)

histories = {}

#%%

for root,subdirs,files in walk(BASEPATH):
    for file in files:
        if "SNR=0" in root:
            pass
        elif file == "history.npy":# and "He=8" in root:
            print(root)

            SNR = retrieve_snr(root)
            HE = retrieve_he(root)
            T = retrieve_T(root)

            history = load(join(root,file),allow_pickle=True)
            history = history.item()
            loss        = history["loss"]
            val_loss    = history["val_loss"]
            name = f"T={T}_He={HE}_SNR={SNR}"
            histories[name]=(loss,val_loss)
            
#%%
            plt.figure(figsize=(16,9),dpi=100)
            plt.title(f"SNR = -{SNR} dB   |   He = {HE}")
            plt.plot(loss,label="Loss")
            plt.plot(val_loss,label=f"Validation Loss")
            plt.xlabel("Number of Epochs")
            plt.ylabel("Mean Squared Error")
            plt.legend()
            plt.ticklabel_format(style="sci",scilimits=(0,2))

#%%
histories = pd.DataFrame(histories)

color_cycle = iter(['C2','C1','C3'])

SNR = 40

plt.figure(figsize=(16,9),dpi=100)

for name, hist in histories.items():
    
    #SNR = retrieve_snr(name)
    HE = retrieve_he(name)
    T = retrieve_T(name)
    
    if f"SNR={SNR}" in name:
        color = next(color_cycle)
        plt.plot(hist[0],color=color,label=f"He = {HE} (Training)")
        plt.plot(hist[1],color=color,marker="x",linestyle="",label=f"He = {HE} (Validation)")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.ticklabel_format(style="sci",scilimits=(0,2))
        
plt.savefig(f"data/plots/history_snr{SNR}.pdf")
#%%
plt.savefig(f"data/plots/history_snr{SNR}.pdf")


# %%
