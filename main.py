#%%
import tensorflow as tf
from numpy import load
from models import TISTA, LISTA
from reader import get_batch
from bernoulli_gaussian import get_bg_batch
from callbacks import *

if __name__ == "__main__":

    TRAINING    = "data/training_100000samples_random_process_04_1372Hz_cankayser.tfrecord"
    VALIDATION  = "data/validation_10000samples_random_process_04_1372Hz_cankayser.tfrecord"
    FILENAME_A  = "data/A_1372Hz.npy"
    FILENAME_W  = "data/W_1372Hz.npy"

    TRAINSIZE   = 100000
    BATCHSIZE   = 200
    LRATE       = 0.0008
    EPOCHS      = 100
    STEPS       = 10

    # A           = load(FILENAME_A)
    # W           = load(FILENAME_W) # W = tf.pinv(A) PENROSE INVERSE
    # train_data  = get_batch(TRAINING,BATCHSIZE)
    # valid_data  = get_batch(VALIDATION,BATCHSIZE)

    A = tf.random.normal((25,50),0, 1.0 / 25**(1/2))
    W = tf.linalg.pinv(A)
    train_data  = get_bg_batch(A,BATCHSIZE)
    valid_data  = get_bg_batch(A,BATCHSIZE)

    callbacks = [tensorboard_cb,checkpoint_cb,early_stopping_cb]
    # callbacks = []

    model   = TISTA(A,W,initial_lambda=0.0,T=15)
    # model   = LISTA(A,initial_lambda=0.0,T=6)

    optim   = tf.keras.optimizers.Adam(LRATE)
    loss    = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optim,loss=loss)

    model.fit(train_data,validation_data=valid_data,batch_size=BATCHSIZE,\
        epochs=EPOCHS,steps_per_epoch=STEPS, validation_steps=STEPS/10, callbacks=callbacks)

#%%
# TEST
import numpy as np
from matplotlib import pyplot as plt 

y,x = next(iter(train_data))
print(np.sum(model.predict(y)[0]))

pred = model.predict(y)[0]
pred = pred[:2601].reshape(51,51)

x = x[0]
x = x[:2601].numpy()
x = x.reshape(51,51)

fig, ax = plt.subplots(2)
ax[0].imshow(x)
ax[1].imshow(pred)
fig.show()





# %%
# Sourcemaps Erstellen model.predict
# 2 Quellen ausm Datensatz rausschmeißen am besten über Indices
# Tensorboard einbdinden