#%%
import tensorflow as tf
from numpy import load
from models import TISTA
from reader import get_batch

if __name__ == "__main__":

    TRAINING    = "data/training_100000samples_random_process_04_1372Hz_cankayser.tfrecord"
    VALIDATION  = "data/validation_10000samples_random_process_04_1372Hz_cankayser.tfrecord"
    FILENAME_A  = "data/A_1372Hz.npy"
    FILENAME_W  = "data/W_1372Hz.npy"

    BATCHSIZE   = 200
    LRATE       = 0.0008
    EPOCHS      = 5
    STEPS       = 100

    train_data  = get_batch(TRAINING,BATCHSIZE)
    valid_data  = get_batch(VALIDATION,BATCHSIZE)
    A           = load(FILENAME_A)
    W           = load(FILENAME_W) # W = tf.pinv(A) PENROSE INVERSE


    model   = TISTA(W)
    optim   = tf.keras.optimizers.Adam(LRATE)
    loss    = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optim,loss=loss)
    model.fit(train_data,validation_data=valid_data,batch_size=BATCHSIZE,\
        epochs=EPOCHS,steps_per_epoch=STEPS)
# %%
