#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from os.path import dirname, join, pardir, normpath
from os import listdir

from tensorflow.keras.models import load_model
from numpy import load

from tools.scratchfiles._jahnke_reverse import transform_tensor_to_sourcemap
from tools.training.loss_funcs import nmse_db
from tools.model import model, A, data
from tools.results.compare_cleansc import tista, cleansc
from tools.environment import SNR, HE, rg, mg 
from tools.training.data import filter_distance, reduce_batchsize

from tools.scratchfiles._evaluate import PlanarSourceMapEvaluator

#tf.config.run_functions_eagerly(True)



NUM_RUNS = 10

PATH = dirname(__file__)
MODELDIR = "models T=30"
    
def get_level_errors(algorithm, data, num_runs=NUM_RUNS):

    data = reduce_batchsize(data)
    data = data.filter(filter_distance)
    data_iter = iter(data)

    eval = PlanarSourceMapEvaluator()
    eval.grid = rg
    
    ole = np.zeros(num_runs)    # OVERALL LEVEL ERROR
    sle = []                    # SPECIFIC LEVEL ERROR
    ile = np.zeros(num_runs)    # INVERSE LEVEL ERROR

    for i in range(num_runs):

        y,x = next(data_iter)
        x_true = transform_tensor_to_sourcemap(x).real
        x_pred = algorithm(y)
        

        smap = transform_tensor_to_sourcemap(x)
        gridlen = smap.shape[0]
        idx = smap.nonzero()

        coordinates = rg.pos()[:2]
        coordinates = coordinates.reshape(gridlen,gridlen,2)

        z_padding = np.expand_dims(np.ones(len(idx[0]))*.5, axis=1)
        eval.target_loc = np.append(coordinates[idx], z_padding, axis=1)
        eval.target_p2 = np.expand_dims(x_true[idx], axis=0)
        eval.sourcemap = np.expand_dims(x_pred, axis=0)

        ole[i] = eval.get_overall_level_error()
        sle.append(eval.get_specific_level_error())
        ile[i] = eval.get_inverse_level_error()
    #sle = np.array(sle)
    return ole, sle, ile

#%%
if __name__ == "__main__":

    errors_tista = get_level_errors(tista, data)
    errors_cleansc = get_level_errors(cleansc, data)

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