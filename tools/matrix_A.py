#%%
import numpy as np
import matplotlib.pyplot as plt
from acoular import RectGrid, MicGeom, SteeringVector
from jahnke import stack_complex_matrix,transform_sv_to_A,find_indices
from config import *

def create_sensing_matrix(freq):

    i = find_indices(mg)
    A = stack_complex_matrix(transform_sv_to_A(sv,freq,i)).astype(np.float32)
    return A

# %%
