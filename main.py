#%%
import tensorflow as tf
from numpy import load
from numpy.linalg import cond
from models import *
from reader import get_batch
from bernoulli_gaussian import get_bg_batch
from callbacks import *
from loss_funcs import *
# import numpy as np


TRAINING    = "data/one_source.tfrecord"
# TRAINING = "data/training_100000samples_random_process_04_1372Hz_cankayser.tfrecord"

BATCHSIZE   = 200
LRATE       = 0.00008
EPOCHS      = 200
STEPS       = 200

# callbacks = [tensorboard_cb,checkpoint_cb,early_stopping_cb]
callbacks = [early_stopping_cb]
# callbacks = []

# filename = "He15.0"
# file = "data/A_" + filename +".npy"
# A = tf.convert_to_tensor(load(file))


if __name__ == "__main__":

            
    for he in [4,8,16]:
        filename = str(he*343)
        file = f"data/A_{he*343}Hz.npy"
        A = tf.convert_to_tensor(load(file))

        for snr in [10,5]:
            train_data = get_bg_batch(A,BATCHSIZE,SNR=snr,noise=True).repeat()
            # train_data  = get_batch(TRAINING,BATCHSIZE).repeat()
            valid_data = train_data

            model   = TISTA(A,initial_lambda=0.0,T=20)
            loss    = tf.keras.losses.MeanSquaredError()
            optim   = tf.keras.optimizers.Adam(LRATE)
            model.compile(optimizer=optim,loss=loss)
            model.fit(train_data,validation_data=valid_data,batch_size=BATCHSIZE,\
                    epochs=EPOCHS,steps_per_epoch=STEPS, validation_steps=STEPS//10,\
                        callbacks=callbacks,verbose=2)

            model.save("models/" + filename)
            model.compile(loss=nmse_db)
            NMSE = model.evaluate(valid_data,steps=10)
# %%
