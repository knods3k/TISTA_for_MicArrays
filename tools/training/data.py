#%%
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import random

from tools.jahnke import transform_graphtensor_to_sourcemap, transform_csm_to_y,\
     find_indices, stack_complex_tensor
from tools.environment import rg, mg 


e = 2.220446e-16
rng = tf.random.Generator.from_seed(1)
I = find_indices(mg)


def rayleigh(shape,sigma=1):
    u = rng.uniform(shape,e,1)
    return sigma * tf.sqrt(-2 * tf.math.log(u))


def collect_random_matrices(A_path):
    As = []
    for (dirpath, dirnames, filenames) in os.walk(A_path):
        for file in filenames:
            path = os.path.join(dirpath, file)
            path = os.path.normpath(path)
            As.append(np.load(path))
    return As


def wis_generator(A,pnz=0.01,SNR=40,noise=True,df=2**10):
    A = tf.convert_to_tensor(A)
    M,N = A.shape

    mask = rng.uniform([N//2],0,1) < pnz
    ray = rayleigh([N//2],1)
    x = tf.where(mask,ray,0)
    #x = tf.abs(x)
    
    paddings = tf.constant([[0,N//2]])
    x = tf.pad(x,paddings)
    
    y = tf.tensordot(A,x,1)

    df *= M
    noise_var = (10**(-SNR/10)) * tf.reduce_mean(tf.abs(y))
    E = (1/df**.5) * noise_var * tf.ones([M,M])
    E = tf.linalg.LinearOperatorFullMatrix(E)
    noise_vector = tfp.distributions.WishartLinearOperator(df, E).sample()
    noise_vector = transform_csm_to_y(noise_vector, I)
    noise_vector = stack_complex_tensor(noise_vector)
    y = y + noise*noise_vector
    #y = y.astype(np.float32)

    yield y,x


def bg_generator(A,pnz=0.01,SNR=40,noise=True):
    A = tf.convert_to_tensor(A)
    M,N = A.shape

    mask = rng.uniform([N//2],0,1) < pnz
    ray = rayleigh([N//2],1)
    x = tf.where(mask,ray,0)
    #x = tf.abs(x)
    
    paddings = tf.constant([[0,N//2]])
    x = tf.pad(x,paddings)
    
    y = tf.tensordot(A,x,1)
    noise_var = (10**(-SNR/10)) * tf.reduce_mean(tf.abs(y))
    y = y + noise*rng.normal([M],0,noise_var)
    #y = y.astype(np.float32)

    yield y,x


def get_dataset(generator,A,**kwargs):
    A = tf.convert_to_tensor(A)
    M,N = A.shape

    output_types    = (tf.float32,tf.float32)
    output_shapes   = ((M),(N))

    call    = lambda: generator(A,**kwargs)
    dataset = tf.data.Dataset.from_generator(call,output_types,output_shapes)
    return dataset


def random_matrix_generator(As, **kwargs):
    A = random.choice(As)
    yield bg_generator(A, **kwargs)



def get_batch(dataset,batchsize):
    dataset = dataset.repeat(-1)
    dataset = dataset.prefetch(-1)
    dataset = dataset.batch(batchsize)
    return dataset


def get_bg_batch(A,batchsize,**kwargs):
    return get_batch(get_dataset(bg_generator,A,**kwargs),batchsize)


def get_wis_batch(A,batchsize,**kwargs):
    return get_batch(get_dataset(wis_generator,A,**kwargs),batchsize)


def get_random_matrix_batch(As, batchsize, **kwargs):
    return get_batch(get_dataset(random_matrix_generator, As, **kwargs))


def reduce_batchsize(dataset,batchsize=1):
    return dataset.unbatch().batch(batchsize)


def filter_distance(_, s, distance=0.):
    x,y,_ = rg.pos()
    smap = transform_graphtensor_to_sourcemap(s)
    idx = tf.where(tf.not_equal(smap,0.))
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x = x[idx[0]]
    y = y[idx[1]]
    mask = tf.square(x) - tf.square(y) > tf.square(distance)
    out = tf.reduce_all(mask)
    return out


    



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