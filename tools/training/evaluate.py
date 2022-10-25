#%%
import tensorflow as tf
import numpy as np
from tools.training.loss_funcs import csm_error, nmse_db

def evaluate_csm_error(data, prediction_function, physics, steps=1):
    data.unbatch().batch(1)
    err = np.empty(steps)
    for i in range(steps):
        y,x = next(iter(data))
        csm_true = physics.vector_to_csm(physics.unstack_complex_vector(y))
        sourcemap = prediction_function(y, physics)
        x_pred = physics.sourcemap_to_vector(sourcemap)
        y_pred =  tf.einsum("ij,kj->ki",physics.A, x_pred)
        csm_pred = physics.vector_to_csm(physics.unstack_complex_vector(y_pred))
        err[i] = csm_error(csm_true, csm_pred).numpy().real
    error = np.mean(err)
    return error


def evaluate_nms_error(data, prediction_function, physics, steps=1):
    data.unbatch().batch(1)
    err = np.empty(steps)
    for i in range(steps):
        y,x = next(iter(data))
        sourcemap_true = physics.vector_to_sourcemap(x)
        sourcemap = prediction_function(y, physics)
        err[i] = nmse_db(sourcemap_true, sourcemap).numpy().real
    error = np.mean(err)
    return error
# %%
