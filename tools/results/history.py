#%%
import numpy as np
from os.path import dirname, join, pardir, normpath, isdir
from os import listdir, walk
from matplotlib import pyplot as plt
from tools.training.convert_models import retrieve_snr, retrieve_he, retrieve_T
import pandas as pd


plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams["figure.dpi"] = 100
plt.rc("font",size=14)

BASEPATH = normpath(join("models","Clean"))

HE = 16
T = 40
MODELNAME = f"He={HE}_T={T}"
HISTORY_PATH = normpath(join(f"{BASEPATH}",f"{MODELNAME}","history.npy"))
history = np.load(HISTORY_PATH,allow_pickle=True)

names = []
histories = []

for root,subdirs,files in walk(BASEPATH):
    for file in files:
        if "SNR=10" in root:
            pass
        elif file == "history.npy":# and "He=8" in root:
            print(root)

            SNR = retrieve_snr(root)
            HE = retrieve_he(root)
            T = retrieve_T(root)
            name = f"T={T}_He={HE}"

            history = np.load(join(root,file),allow_pickle=True)
            history = history.item()
            history = pd.DataFrame(history)
            # name = file
            names.append(name)
            histories.append(history)
histories = pd.concat(histories, keys=names, axis=1)

color_cycle = iter(['C2','C1','C3'])

plt.figure(figsize=(16,9),dpi=100)

for name, loss in histories:
    T = retrieve_T(name)
    histories[name] = histories[name][histories[name].apply(\
        lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]

    # histories[name].dropna(inplace=True)

    if loss == "loss":
        color = next(color_cycle)

        plt.plot(histories[name].dropna()["loss"], color=color, label=f"Training T={T}")
        plt.plot(histories[name].dropna()["val_loss"],\
            color=color, marker="x", linestyle="", label=f"Validation")
        plt.xlabel("Number of Epochs")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.ticklabel_format(style="sci",scilimits=(0,2))

#%%     
plt.savefig(f"data/plots/history_snr{SNR}.pdf")


# %%
