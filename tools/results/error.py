#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os.path import dirname, join, pardir, normpath
from os import listdir

from tensorflow.keras.models import load_model
from numpy import load

from tools.scratchfiles._jahnke_reverse import transform_tensor_to_sourcemap
from tools.training.loss_funcs import nmse_db
from tools.model import model, A, data
from tools.results.compare_cleansc import tista, cleansc
from tools.environment import SNR, HE 



num_runs = 1000

PATH = dirname(__file__)
MODELDIR = "models T=30"

#%%
if __name__ == "__main__":

    data_iter = iter(data)

    errors_tista = np.zeros(num_runs)
    errors_cleansc = np.zeros(num_runs)

    for i in range(num_runs):

        y,x = next(data_iter)

        x_true = transform_tensor_to_sourcemap(x).real

        x_tista = tista(y)[0]
        x_cleansc = cleansc(y)[0]

        errors_tista[i] = nmse_db(x_tista,x_true)
        errors_cleansc[i] = nmse_db(x_cleansc,x_true)

    errors_tista = {"model": "TISTA", "SNR": SNR, "HE": HE, "errors": errors_tista}
    errors_cleansc = {"model": "CLEAN-SC", "SNR": SNR, "HE": HE, "errors": errors_cleansc}
    errors = pd.DataFrame([errors_tista, errors_cleansc])
    errors.to_pickle(f"data/errors/SNR={SNR}_HE={HE}.pkl")

#%%
if __name__ == "__main__":
    ERRORDIR = normpath(join(PATH,pardir,pardir,"data","errors"))
    dirs = listdir(ERRORDIR)
    errors = pd.DataFrame()
    for d in dirs:
        try:
            print("\n"*3,d)
            d = normpath(join(ERRORDIR, d))
            pickle = pd.read_pickle(d)
            SNR = pickle["SNR"][1]
            HE = pickle["HE"][1]
            MODEL = pickle["model"][1]
            err = pd.DataFrame(pickle[pickle["model"] == "CLEAN-SC"]["errors"][1])
            errors[f"{MODEL}_HE={HE}_SNR={SNR}"] = err[np.isfinite(err).all(1)]
            print(errors.describe(percentiles=[]).drop("count").drop("50%").apply(np.round,decimals=2).T[["min","max","mean","std"]].to_latex())
            

        except:
            pass



# %%