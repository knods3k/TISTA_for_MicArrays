#%%
from tensorflow.keras.models import load_model
from os.path import dirname, join, pardir, normpath
from os import listdir
from shutil import rmtree
from tools.training.loss_funcs import nmse_db, nmse
from tools.training.data import DataGenerator
from tools.physical import PhyiscalModel
from tools.environment import INCREMENT, NMICS
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


plt.rc("font",size=14)

#%%
NSOURCES = 10

all = []

for HE in [4,8,16]:
    freq = HE*343
    MODELDIR = normpath(join(f"models\convergence\He={HE}"))


    dirs = listdir(MODELDIR)
    dirs.sort()

    errors = []
    Ts = []
    for d in dirs:
        try:
            print(d)
            modelpath = normpath(join(MODELDIR,d))
            model = load_model(modelpath)
            A = model.A_save
            T = model.T_save.numpy()
            pnz = NSOURCES / (A.shape[1] // 2)
            generator = DataGenerator(A,100,pnz=pnz, noise=False)
            data = generator.get_batch().repeat()
            model.compile(loss=nmse)
            error = model.evaluate(data,steps=1, batch_size=100)
            Ts.append(T)
            errors.append(error)
        except (FileNotFoundError, OSError):
            try:
                rmtree(modelpath)
                print(f"Removed {modelpath}")
            except NotADirectoryError:
                pass
        except (NotADirectoryError):
            pass

    all.append(pd.DataFrame({'T': Ts, str(HE): errors}).set_index('T'))

pd.concat(all, axis=1)

#%%    

errors = pd.concat(errors, ignore_index=True, axis=1)
errors



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