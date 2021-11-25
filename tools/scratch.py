#%%
from keras.models import load_model
from models import TISTA
from config import A,data,model

PATH = "scratch/model"

#model = TISTA(A)
model.compile()

model.predict(next(iter(data))[0])
model.save_weights(PATH)
# %%
model = TISTA(A)
model.load_weights(PATH)
# %%
