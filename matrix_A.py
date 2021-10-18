#%%
import numpy as np
import matplotlib.pyplot as plt
from random_process_01 import BLOCKSIZE, REF_MIC, ps, rg, mg, sm
from jahnke import stack_complex_matrix, stack_complex_vector,transform_rg_mg_to_A,find_indices
from helper import get_p2_at_reference, get_csm
# from main import REF_MIC, BLOCKSIZE

FREQ = 343 * 16
fidx = ps.fftfreq().searchsorted(FREQ)

i = find_indices(mg)
A = stack_complex_matrix(transform_rg_mg_to_A(rg,mg,FREQ,i)).astype(np.float32)

np.save("A_" + str(FREQ) + "Hz",A)

#%%
import cv2 as cv
A = np.load("A_" + str(FREQ) + "Hz.npy")
win = cv.namedWindow("image",flags=cv.WINDOW_GUI_NORMAL)
cv.imshow("image", A)
cv.waitKey(1)


# %%
