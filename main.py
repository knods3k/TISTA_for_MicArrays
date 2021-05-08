#%%
import tensorflow as tf
from numpy import load
from numpy.linalg import cond
from models import *
from reader import get_batch
from bernoulli_gaussian import get_bg_batch
from callbacks import *
from loss_funcs import *

if __name__ == "__main__":

    TRAINING    = "data/training_100000samples_random_process_04_1372Hz_cankayser.tfrecord"
    VALIDATION  = "data/validation_10000samples_random_process_04_1372Hz_cankayser.tfrecord"
    FILENAME_A  = "data/A_1372Hz.npy"
    # FILENAME_A  = "data/A_5488Hz.npy"
    # FILENAME_A  = "data/A_random.npy"
    # FILENAME_W  = "data/W_1372Hz.npy"

    TRAINSIZE   = 100000
    BATCHSIZE   = 200
    LRATE       = 0.00008
    EPOCHS      = 100
    STEPS       = 100

    A = tf.convert_to_tensor(load(FILENAME_A))
    # W = tf.linalg.pinv(A)

    # train_data  = get_batch(TRAINING,BATCHSIZE)
    # valid_data  = get_batch(VALIDATION,BATCHSIZE)

    # A = tf.random.normal((M,N),0, 1.0 / N**(1/2))

    # callbacks = [tensorboard_cb,checkpoint_cb,early_stopping_cb]
    callbacks = [early_stopping_cb]
    # callbacks = []

    filenames = ["1372Hz","2744Hz","5488Hz","random"]
    info = []
    for filename in filenames:
        file = "data/A_" + filename +".npy"
        A = tf.convert_to_tensor(load(file))
        train_data  = get_bg_batch(A,BATCHSIZE,pnz=0.01,SNR=40,noise=False)
        valid_data  = train_data
        condition = cond(A)
        model   = TISTA(A,initial_lambda=0.0,T=20)
        loss    = tf.keras.losses.MeanSquaredError()

        optim   = tf.keras.optimizers.Adam(LRATE)
        model.compile(optimizer=optim,loss=loss)
        model.fit(train_data,validation_data=valid_data,batch_size=BATCHSIZE,\
            epochs=EPOCHS,steps_per_epoch=STEPS, validation_steps=STEPS/10,\
                callbacks=callbacks,verbose=1)

        model.compile(loss=nmse_db)
        NMSE = model.evaluate(valid_data,steps=10)
        info.append([filename,condition,NMSE])

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