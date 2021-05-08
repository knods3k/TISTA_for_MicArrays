import tensorflow as tf
import numpy as np

def bg_generator(A,pnz=0.01,SNR=40,noise=True):
    M,N = A.shape
    noise_var = pnz*N/M * (10 ** (-SNR/10))

    x = ((np.random.uniform( 0,1,N)<pnz) * np.random.normal(0,1,N)).astype(np.float32)
    x = x**2
    y = (np.matmul(A,x) + noise*np.random.normal(0,np.sqrt(noise_var),M)).astype(np.float32)

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