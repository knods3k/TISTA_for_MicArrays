#%%

#%%
from os.path import dirname, join, pardir, normpath
from os import getcwd
import time 
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras.models import load_model
from numpy import save, load
from tools.training.models import TISTA
from tools.training.data import DataGenerator
from tools.training.callbacks import early_stopping_cb,checkpoint_cb,tensorboard_cb
from tools.training.loss_funcs import nmse_db
from tools.physical import PhyiscalModel
#from tools.matrix_A import create_sensing_matrix

BATCHSIZE   = 200
LRATE       = 0.0008
EPOCHS      = 200
STEPS       = 200
VERBOSITY   = 1

NMICS = 64
INCREMENT = 0.01
NSOURCES = 10

Ts = range(60,121,5) #[20,40,60]
SNRs = [40]
HEs = [4,8]

SESSION = "Clean"
MODELDIR = normpath(join("models", f"{SESSION}"))

#callbacks = [tensorboard_cb,checkpoint_cb,early_stopping_cb]
#callbacks = [checkpoint_cb,early_stopping_cb]
callbacks = [early_stopping_cb]
# callbacks = []

if __name__ == "__main__":

    for he in HEs:
        for T in Ts:
            physics = PhyiscalModel(he, INCREMENT, NMICS)
            random_matrix_path = normpath(join("data", "random_matrices", f"{he}"))
            As = random_matrix_path
            # assert A.shape == As[0].shape
            A = physics.A
            pnz = NSOURCES / (A.shape[1] // 2)

            modelname = f"He={he}_T={T:02d}"

            normal = tfd.Normal
            wishart = tfd.WishartLinearOperator
            generator = DataGenerator(A,BATCHSIZE,pnz=pnz, noise=False)
            train_data = generator.get_batch().repeat()
            #noisy_generator = DataGenerator(As,BATCHSIZE,pnz=pnz, noise=False)
            #valid_data = noisy_generator.get_batch().repeat()
            valid_data = train_data
#%%
            model   = TISTA(A,T=T)
            #loss    = tf.keras.losses.MeanSquaredError()
            loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
            optim   = tf.keras.optimizers.Adam(LRATE)
            model.compile(optimizer=optim,loss=loss)
            history = model.fit(train_data,validation_data=valid_data,batch_size=BATCHSIZE,\
                    epochs=EPOCHS,steps_per_epoch=STEPS, validation_steps=STEPS//10,\
                        callbacks=callbacks,verbose=VERBOSITY)
            savepath = normpath(join(MODELDIR, modelname))
            print(f"Saving model at {join(getcwd(),savepath)}")
            model.save(savepath)
            save(normpath(join(savepath,"history.npy")),history.history)
            #save(normpath(join(MODELDIR,modelname,f"A_{HE}.npy")),A)

            model.compile(loss=nmse_db)
            NMSE = model.evaluate(valid_data,steps=10)
            with open("logfile.txt","a") as f:
                print(f"{modelname}: NMSE_dB = {NMSE} \t {time.asctime()}",file=f)


#%%

if False:
# INITIALLY TRAIN MODELS
    for HE in [16,8,4]:
        for SNR in [40,20,10]:
            A = create_sensing_matrix(HE*343)
            A = tf.convert_to_tensor(A)

            modelname = f"He={HE}_SNR={SNR}"

            train_data = get_bg_batch(A,BATCHSIZE,SNR=SNR,noise=True).repeat()
            valid_data = train_data

            model   = TISTA(A,T=30)
            loss    = tf.keras.losses.MeanSquaredError()
            optim   = tf.keras.optimizers.Adam(LRATE)
            model.compile(optimizer=optim,loss=loss)
            history = model.fit(train_data,validation_data=valid_data,batch_size=BATCHSIZE,\
                    epochs=EPOCHS,steps_per_epoch=STEPS, validation_steps=STEPS//10,\
                        callbacks=callbacks,verbose=VERBOSITY)

            model.save("models/" + modelname)
            save(f"models/{modelname}/history.npy",history.history)
            save(f"models/{modelname}/A_{HE}.npy",A)

            model.compile(loss=nmse_db)
            NMSE = model.evaluate(valid_data,steps=10)
            print(f"{modelname}: NMSE_dB = {NMSE}")

#%%
# REFINE EXISTING MODELS
    for HE in [16,8,4]:
        for SNR in [60,40,20]:
            A = load(f"models/He={HE}_SNR={SNR}/A_{HE}.npy")

            train_data = get_bg_batch(A,BATCHSIZE,SNR=SNR,noise=True).repeat()
            valid_data = train_data

            modelname = f"He={HE}_SNR={SNR}"
            model = load_model(f"models/He={HE}_SNR={SNR}")
            loss    = tf.keras.losses.MeanSquaredError()
            optim   = tf.keras.optimizers.Adam(LRATE*0.05)
            model.compile(optimizer=optim,loss=loss)
            history = model.fit(train_data,validation_data=valid_data,batch_size=BATCHSIZE,\
                    epochs=EPOCHS,steps_per_epoch=STEPS, validation_steps=STEPS//10,\
                        callbacks=callbacks,verbose=VERBOSITY)
            model.save("models/" + modelname)
            save(f"models/{modelname}/history.npy",history.history)
            model.compile(loss=nmse_db)
            NMSE = model.evaluate(valid_data,steps=10)
#%%
                
# %%
