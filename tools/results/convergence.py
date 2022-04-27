#%%
from tensorflow.keras.models import load_model
from os.path import dirname, join, pardir, normpath
from os import listdir
from tools.training.loss_funcs import nmse_db
from tools.training.data import get_bg_batch
from tools.jahnke import create_sensing_matrix
from tools.environment import HE
from matplotlib import pyplot as plt
import numpy as np

plt.rc("font",size=14)


he_dict = {}

for HE in [16]:
    freq = HE*343

    PATH = dirname(__file__)
    MODELDIR = normpath(join(PATH,pardir,pardir,f"models_64_T=[1,30]"))
    A = create_sensing_matrix(freq)
    data = get_bg_batch(A,1,SNR=999,noise=False)

    dirs = listdir(MODELDIR)
    dirs.sort()
    errors = []

    for d in dirs:
        print(d)
        modelpath = normpath(join(PATH,pardir,MODELDIR,d))
        model = load_model(modelpath,compile=True)
        #model.compile(loss=nmse_db)
        e = model.evaluate(data,steps=100)
        errors.append(e)
#%%
    he_dict[f"{HE}"] = errors[:]

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
for he, err in he_dict.items():
    ax.plot(err, label=f"He = {he}")
ax.legend()
ax.set_xlabel("Number of Layers")
ax.set_ylabel("Mean Squared Error")
fig.savefig("data/plots/convergence.pdf")
fig.show()
fig.savefig(MODELDIR+"/convergence.pdf")


    

#%%
plt.figure(figsize=(16,9),dpi=100)
plt.plot(errors)
plt.xlabel("Number of Layers")
plt.ylabel("Mean Squared Error")
plt.savefig(MODELDIR+"/convergence.pdf")

#%%
from scipy.optimize import curve_fit
from scipy import interpolate

i = np.arange(len(errors))
x = np.arange(len(errors)*2)

def exp(x, a, b, c):
    return a* np.exp(-b * x) + c

popt, pcov = curve_fit(exp, i, errors)



plt.figure()
plt.plot(i, errors, label="original data")
plt.plot(x, exp(x, *popt), 'r-', label="Fitted Curve")
plt.legend()
plt.show()
# %%