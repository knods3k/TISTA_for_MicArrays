#%%
from tensorflow.keras.models import load_model
from os.path import dirname, join, pardir, normpath
from os import listdir
from tools.training.loss_funcs import nmse_db
from config import data
from matplotlib import pyplot as plt

PATH = dirname(__file__)
MODELDIR = normpath(join(PATH,pardir,"models T=[3,30]"))

dirs = listdir(MODELDIR)
dirs.sort()
errors = []

for d in dirs[0:1]:
    print(d)
    modelpath = normpath(join(PATH,pardir,MODELDIR,d))
    model = load_model(modelpath,compile=True)
    #model.compile(loss=nmse_db)
    e = model.evaluate(data,steps=100)
    errors.append(e)

plt.figure(figsize=(16,9),dpi=100)
plt.plot(errors)
plt.savefig(MODELDIR+"/convergence.png")
# %%