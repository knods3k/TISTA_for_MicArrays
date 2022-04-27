#%%
from os.path import dirname,pardir,normpath,join

from matplotlib import pyplot as plt 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


from acoular import RectGrid, MicGeom, L_p
from tools.scratchfiles._evaluate import PlanarSourceMapEvaluator
from tools.training.data import get_bg_batch
from tools.jahnke import transform_rg_mg_to_A, find_indices, transform_tensor_to_sourcemap, stack_complex_matrix


HE = 16
SNR = 999

def retrieve_test_sourcemap(micgeom_path,increment,model):
    mg = MicGeom(from_file=micgeom_path)
    rg = RectGrid(x_max=0.5,x_min=-0.5,y_max=0.5,y_min=-0.5,z=0.5,increment=increment)
    A = transform_rg_mg_to_A(rg,mg,343*HE,find_indices(mg))
    A = stack_complex_matrix(A)
    A = tf.convert_to_tensor(A)

    S1 = {"x": 0.3, "y": 0.1, "p2": 2}
    S2 = {"x": -0.12, "y": 0.2, "p2": 1.4}
    S3 = {"x": 0.32, "y": -0.23, "p2": 1}

    target_loc = []
    target_p2 = []
    x = np.zeros(rg.shape)
    for S in [S1,S2,S3]:
        i,j,k = S["x"], S["y"], rg.z
        p2 = S["p2"]
        x[rg.index(i,j)] = p2
        target_loc.append([i,j,k])
        target_p2.append(p2)

    x = tf.convert_to_tensor(x.flatten())
    paddings = tf.constant([[0,x.shape[0]]])
    x = tf.pad(x,paddings)
    x = tf.expand_dims(x, axis=0)

    y = tf.einsum("ij,kj->ki",A,x)
    
    x_pred = model(y)

    smap_true = transform_tensor_to_sourcemap(x)
    smap_pred = transform_tensor_to_sourcemap(x_pred)

    eval = PlanarSourceMapEvaluator()
    eval.grid = rg
    eval.target_loc = np.array(target_loc)
    eval.target_p2 = np.array([target_p2])
    eval.sourcemap = np.array([smap_pred])



    return smap_true, smap_pred, eval
    
model_64 = load_model(normpath(join("models_64_T=[1,30]","He=16_SNR=999_T=30")))
smap_hi, smap_pred_hi, evaluator_hi = retrieve_test_sourcemap("data/_tub_vogel64_ap1.xml", 0.01, model_64)

model_16 = load_model(normpath(join("models He=16","He=16_SNR=999_T=11")))
smap_lo, smap_pred_lo, evaluator_lo = retrieve_test_sourcemap("data/_tub_vogel16_ap1.xml", 0.04, model_16)


# %%
count = 1
plt.subplots(figsize=(16,9),sharey=True,sharex=True)
for smap in [smap_hi,smap_pred_hi,smap_lo,smap_pred_lo]:
    plt.subplot(3,2,count)
    plt.imshow(L_p(smap.T),origin="lower",vmax=95,vmin=75,extent=[-.5,.5,-.5,.5])
    count +=1
plt.colorbar()
# %%
