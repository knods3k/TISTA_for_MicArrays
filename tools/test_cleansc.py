#%%
from _spectra_import import PowerSpectraImport
from _evaluate import PlanarSourceMapEvaluator
from bernoulli_gaussian import get_bg_batch
from _jahnke_reverse import transform_y_to_csm, unstack_complex_vector, reshape_sourcemap
from traits.api import CArray
from matrix_A import rg 
from numpy import load, where, expand_dims
from tensorflow.keras.models import load_model
from acoular.fbeamform import BeamformerCleansc
from config import SNR, HE, model, A, data

y_,x_ = next(iter(data))

y = unstack_complex_vector(y_[0].numpy())
x_true = unstack_complex_vector(x_[0].numpy())
x_pred = model(y_)
x_pred = unstack_complex_vector(x_pred[0].numpy())

csm = transform_y_to_csm(y)
csm = expand_dims(csm,axis=0)
srcmap_true = reshape_sourcemap(x_true).real
srcmap_pred = reshape_sourcemap(x_pred).real

ps = PowerSpectraImport()
ps._set_csm(csm)

ev = PlanarSourceMapEvaluator()
ev.sourcemap = expand_dims(srcmap_pred,0)
ev.grid = rg

plane = rg.pos()
plane_coords = plane.reshape(26,26,3)
idx = where(srcmap_true != 0)
src_pos = plane_coords[idx]
src_pow = srcmap_true[idx]
src_pow = expand_dims(src_pow,0)
ev.target_loc = src_pos
ev.target_p2 = src_pow

err = ev.get_inverse_level_error()

print(err)

bf = BeamformerCleansc()







# %%



# %%
