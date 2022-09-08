#%%
from os.path import normpath, join
import acoupipe
import numpy as np
from scipy.stats import norm
from tools.physical import PhyiscalModel

INCREMENT = 0.01
NMICS = 64
DIR = normpath(join("data","random_matrices"))


for HE in [4, 8, 16]:
    for i in range(0,100):
        physics = PhyiscalModel(HE,INCREMENT,NMICS)
        mgs = acoupipe.sampler.MicGeomSampler()
        mgs.target = physics.mg

        pos = physics.mg.mpos
        distance = np.abs(pos - pos[:,None])
        min_distance = np.min(distance[distance!=0])
        scale = min_distance * (1/3)
        mgs.random_var = norm(scale=scale)
        mgs.ddir[0] = 1.
        mgs.ddir[1] = 1.


        name = f"A_{i:02d}.npy"
        path = normpath(join(DIR,f"{HE}",name))
        mgs.sample()
        np.save(path, physics.A)

#%%
for i in range(2):
    name = f"A_{i:02d}.npy"
    path = normpath(join(DIR,f"{HE}",name))
    A = np.load(path)
    #mgs.sample()
    print(np.mean(A - physics.A))
    print(A.dtype)



# %%
