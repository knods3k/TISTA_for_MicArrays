#%%
from tensorflow.keras.models import load_model
from os.path import dirname, join, pardir, normpath
from os import listdir
from shutil import rmtree
from tools.training.loss_funcs import nmse_db, nmse
from tools.training.data import DataGenerator
from tools.physical import PhyiscalModel
from tools.environment import INCREMENT, NMICS
from tools.training.evaluate import evaluate_csm_error, evaluate_nms_error
from tools.prediction_models import tista, cmf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


plt.rc("font",size=14)

import warnings
warnings.filterwarnings('ignore')

#%%
NSOURCES = 10
NRUNS = 10 # NUMBER OVER TESTCASES

all = []

for model_key in ['TISTA', 'CMF']:
    for HE in [4, 16]:
        freq = HE*343
        modeldir = normpath(join(f"models\convergence\He={HE}"))
        physics = PhyiscalModel(HE, INCREMENT, NMICS)

        random_matrix_path = normpath(join("data", "random_matrices", f"{HE}"))
        As = random_matrix_path


        dirs = listdir(modeldir)
        dirs.sort()

        errors = []
        Ts = []
        for d in dirs:
            try:    
                print(d)
                modelpath = normpath(join(modeldir,d))
                model = load_model(modelpath)
                A = model.A_save
                T = model.T_save.numpy()
                Ts.append(T)

                generator = DataGenerator(As,1,nsources=NSOURCES, noise=True)
                data = generator.get_batch().repeat()

                tista_predict = lambda *args: tista(*args, T=T)
                cmf_predict = lambda *args: cmf(*args, max_iter=T)

                if model_key == 'TISTA':
                    error = evaluate_nms_error(data, tista_predict, physics, NRUNS)
                elif model_key == 'CMF':
                    error = evaluate_nms_error(data, cmf_predict, physics, NRUNS)
                errors.append(error)
            except (FileNotFoundError, OSError):
                try:
                    rmtree(modelpath)
                    print(f"Removed {modelpath}")
                except NotADirectoryError:
                    pass
            except (NotADirectoryError):
                pass


        df = pd.DataFrame({'Iterations': Ts, f'He={HE}': errors})
        df = df.set_index('Iterations')
        df.columns = pd.MultiIndex.from_product([[f'{model_key}'], df.columns])
        all.append(df)

results = pd.concat(all, axis=1)

#%%    

errors = pd.concat(errors, ignore_index=True, axis=1)
errors

#%%
plt.plot(df_tista['TISTA']['Iterations'], df_tista['TISTA']['He=16'], label='TISTA')
plt.plot(df_cmf['CMF']['Iterations'], df_cmf['CMF']['He=16'], label='CMF')
plt.xlabel('Number of Layers/Iterations')
plt.ylabel('Normalized CSM Reconstruction Error')
plt.title('He=16')
plt.legend()


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
fig.savefig(modeldir+"/convergence.pdf")


    

#%%
plt.figure(figsize=(16,9),dpi=100)
plt.plot(errors)
plt.xlabel("Number of Layers")
plt.ylabel("Mean Squared Error")
plt.savefig(modeldir+"/convergence.pdf")

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