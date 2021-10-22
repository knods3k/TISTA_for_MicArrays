#%%
import tensorflow as tf
from numpy import save
from tools.models import TISTA
from tools.bernoulli_gaussian import get_bg_batch
from tools.callbacks import *
from tools.loss_funcs import *
from tools.matrix_A import create_sensing_matrix

BATCHSIZE   = 200
LRATE       = 0.008
EPOCHS      = 200
STEPS       = 50

# callbacks = [tensorboard_cb,checkpoint_cb,early_stopping_cb]
callbacks = [early_stopping_cb]
# callbacks = []

if __name__ == "__main__":

            
    for he in [16,8,4]:
        A = create_sensing_matrix(he*343)
        A = tf.convert_to_tensor(A)

        for snr in [60,40,20]:
            modelname = "He="+str(he)+"_SNR="+str(snr)
            save(f"models/{modelname}/A_{he}.npy",A)

            train_data = get_bg_batch(A,BATCHSIZE,SNR=snr,noise=True).repeat()
            valid_data = train_data

            print(f"SNR = {snr}")
            print(f"He = {he}")

            model   = TISTA(A,T=20)
            loss    = tf.keras.losses.MeanSquaredError()
            optim   = tf.keras.optimizers.Adam(LRATE)
            model.compile(optimizer=optim,loss=loss)
            history = model.fit(train_data,validation_data=valid_data,batch_size=BATCHSIZE,\
                    epochs=EPOCHS,steps_per_epoch=STEPS, validation_steps=STEPS//10,\
                        callbacks=callbacks,verbose=1)

            model.save("models/" + modelname)
            model.compile(loss=nmse_db)
            save(f"models/{modelname}/history.npy",history.history)
            NMSE = model.evaluate(valid_data,steps=10)
# %%
