#%%
from os.path import normpath, join

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from tools.training.data import DataGenerator
from tools.training.models import eta
from tools.training.loss_funcs import nmse_db, nmse, csm_error
from tools.physical import PhyiscalModel
from tools.environment import NMICS, INCREMENT

from tools.results.compare_cleansc import cleansc, tista, cmf

def evaluate_csm_error(data, prediction_function, physics, steps=1):
        err = np.empty(steps)
        for i in range(steps):
            y,x = next(iter(data))
            csm_true = physics.vector_to_csm(physics.unstack_complex_vector(y))
            sourcemap = prediction_function(y)
            x_pred = physics.sourcemap_to_vector(sourcemap)
            y_pred =  tf.einsum("ij,kj->ki",physics.A, x_pred)
            csm_pred = physics.vector_to_csm(physics.unstack_complex_vector(y_pred))
            err[i] = csm_error(csm_true, csm_pred).numpy().real
        error = np.mean(err)
        return error


T = 60
NRUNS = 40  # NUMBER OF EVALUATION STEPS

models = {'TISTA': tista, 'CMF': cmf}

#%%
all = []
for model_key in ['TISTA', 'CMF']:
    for HE in [4, 16]:

        physics = PhyiscalModel(HE, INCREMENT, NMICS)
        A = physics.A

        # MODELDIR = normpath(join(f"models\convergence\He={HE}\He={HE}_T={T}"))
        # model = load_model(MODELDIR)

        random_matrix_path = normpath(join("data", "random_matrices", f"{HE}"))
        As = random_matrix_path

        errors = []
        source_amounts = []
        for nsources in range (10, 201, 10):
            pnz = nsources / (A.shape[1] // 2)
            data = DataGenerator(As, 1, pnz).get_batch().repeat()


            data = data.filter(lambda _,x: tf.math.count_nonzero(x) == nsources)

            model = models[model_key]
            error = evaluate_csm_error(data, model, physics, 1)

            # err = np.empty(NRUNS)
            # for i in range(NRUNS):
            #     y,x = next(iter(data))
            #     csm_true = physics.vector_to_csm(physics.unstack_complex_vector(y))
            #     x_pred = model.call(y)
            #     y_pred =  tf.einsum("ij,kj->ki",A, x_pred)
            #     csm_pred = physics.vector_to_csm(physics.unstack_complex_vector(y_pred))
            #     err[i] = csm_error(csm_true, csm_pred).numpy().real
            # error = np.mean(err)
            # print(f'Number of Sources: {nsources}, Error: {error}')

            errors.append(error)
            source_amounts.append(nsources)

        df = pd.DataFrame({'Number of Sources': source_amounts, f'He={HE}': errors})
        df = df.set_index('Number of Sources')
        df.columns = pd.MultiIndex.from_product([[f'{model_key}'], df.columns])
        all.append(df)


    results = pd.concat(all, axis=1)
    # results.to_pickle('')







# %%
