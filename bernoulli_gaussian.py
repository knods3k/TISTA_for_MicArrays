#%%
import tensorflow as tf
import numpy as np

def bg_generator(A,pnz=0.01,SNR=40,noise=True):
    M,N = A.shape
    # noise_var = (10 ** (-SNR/10))

    x = ((np.random.uniform( 0,1,N)<pnz*2) * np.random.normal(0,1,N))
    x = np.abs(x)
    x[N//2:] = 0
    x = x.astype(np.float32)
    # x[:] = 1
    
    y = np.matmul(A,x)
    # noise_var = np.sqrt((10 ** (-SNR/10)) * np.mean(np.abs(y))**2 * 1.825) #(N/M)**(pnz**(M/N)))
    noise_var = (10**(-SNR/10)) * np.mean(np.abs(y))
    y = y + noise*np.random.normal(0,noise_var,M)
    y = y.astype(np.float32)

    yield y,x

def bg_dataset(generator,A,**kwargs):
    M,N = A.shape

    output_types    = (tf.float32,tf.float32)
    output_shapes   = ((M),(N))

    call    = lambda: generator(A,**kwargs)
    dataset = tf.data.Dataset.from_generator(call,output_types,output_shapes)
    return dataset

def bg_batch(dataset,batchsize):
    dataset = dataset.repeat(-1)
    dataset = dataset.prefetch(-1)
    dataset = dataset.batch(batchsize)
    return dataset

def get_bg_batch(A,batchsize,**kwargs):
    return bg_batch(bg_dataset(bg_generator,A,**kwargs),batchsize)

# filenames = ["1372Hz"]
# info = []
# for SNR in [60,40,20,0]:    
#     for filename in filenames:
#         file = "data/A_" + filename +".npy"
#         A = tf.convert_to_tensor(np.load(file))
#         train_data  = get_bg_batch(A,400,pnz=0.005,SNR=SNR,noise=True)
#         valid_data  = train_data
#         y,x = next(iter(train_data))
        
#         y_true = tf.einsum("ij,kj->ki",A,x)
#         signal = tf.reduce_mean(tf.abs(y_true)).numpy()
#         noise = tf.reduce_mean(tf.abs(y_true - y)).numpy()
#         ratio = signal / noise
#         log = 10*np.log10(ratio)
#         print(SNR,log)

# %%
