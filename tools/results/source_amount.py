#%%
from os.path import normpath, join

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from tools.training.data import DataGenerator
from tools.training.models import eta
from tools.training.loss_funcs import nmse_db, nmse, csm_error
from tools.training.evaluate import evaluate_nms_error, evaluate_csm_error
from tools.physical import PhyiscalModel
from tools.environment import NMICS, INCREMENT

from tools.prediction_models import cleansc, tista, cmf
# import warnings
# warnings.filterwarnings('ignore')

# TODO: ADD TO PLOT CMF WITH 60 ITERATIONS
# TODO: CALCULATE NMSE FOR CM


T = 60
NRUNS = 10  # NUMBER OF EVALUATION STEPS

#models = {'TISTA': tista, 'CMF': cmf}

#%%
all = []
for model_key in ['TISTA','CMF']:
    for HE in [4, 16]:

        physics = PhyiscalModel(HE, INCREMENT, NMICS)
        A = physics.A

        MODELDIR = normpath(join(f"models\convergence\He={HE}\He={HE}_T={T}"))
        model = load_model(MODELDIR)

        random_matrix_path = normpath(join("data", "random_matrices", f"{HE}"))
        As = random_matrix_path

        errors = []
        source_amounts = []
        for nsources in range (10, 101, 10):
            data = DataGenerator(As, 1, nsources).get_batch().repeat()


            data = data.filter(lambda _,x: tf.math.count_nonzero(x) == nsources)


            if model_key == 'TISTA':
                error = evaluate_nms_error(data, tista, physics, NRUNS)
            elif model_key == 'CMF':
                error = evaluate_nms_error(data, cmf, physics, NRUNS)

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
