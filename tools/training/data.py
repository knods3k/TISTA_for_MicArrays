#%%
#%%
from posixpath import normpath
from importlib_metadata import distribution
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
import os
import random

from tools.physical import physics
from tools.environment import rg, mg


A_path = os.path.normpath(os.path.join("data","random_matrices","16"))


def collect_random_matrix_path(A_path):
    As = []
    for (dirpath, dirnames, filenames) in os.walk(A_path):
        for file in filenames:
            path = os.path.join(dirpath, file)
            path = os.path.normpath(path)
            try:
                As.append(path)
            except MemoryError:
                break
    return As

class DataGenerator():
    def __init__(self, A, batchsize=1, pnz=0.01, SNR=40, noise=True, df=1, distribution=tfd.Normal):
        if type(A) is str:
            self.As = collect_random_matrix_path(A)
            self.A = np.load(self.As[1])
            self.A = tf.convert_to_tensor(self.A)

        else:
            self.As = None
            self.A = tf.convert_to_tensor(A)
        self.M, self.N = self.A.shape
        self.I = np.triu_indices(self.M//2)
        self.batchsize = batchsize
        self.pnz = pnz
        self.SNR = SNR
        self.noise = noise
        self.df = df
        self.distribution = distribution
        self.rng = tf.random.Generator.from_seed(1)
        self.e = 2.220446e-16

    def __iter__(self):
        return
    

    def rayleigh(self, shape, sigma=1):
        u = self.rng.uniform(shape, self.e, 1)
        return sigma * tf.sqrt(-2 * tf.math.log(u))

    
    def generate(self):
        N_vec = tf.ones(self.N // 2)
        mask = tfd.Uniform(0*N_vec, 1*N_vec).sample() < self.pnz
        ray = self.rayleigh([self.N//2],1)
        x = tf.where(mask, ray, 0)
        paddings = tf.constant([[0,self.N//2]])
        x = tf.pad(x,paddings)
        y = tf.tensordot(self.A,x,1)

        noise_var = (10**(-self.SNR/10)) * tf.reduce_mean(tf.abs(y))

        if self.noise is False:
            pass

        elif distribution is tfd.Normal:
            y = distribution(y, noise_var)
        
        elif distribution is tfd.WishartLinearOperator:
            df *= self.M
            E = (np.pi/df**.5) * noise_var**.5 * tf.eye(self.M,self.M)
            E = tf.linalg.LinearOperatorFullMatrix(E)
            noise_vector = tfd.WishartLinearOperator(df, E).sample()
            noise_vector = PhyiscalModel.csm_to_vector(noise_vector, self.I)
            noise_vector = PhyiscalModel.stack_complex_vector(noise_vector)

            y += noise_vector

        yield y, x

    @property
    def dataset(self):
        output_types    = (tf.float32,tf.float32)
        output_shapes   = ((self.M),(self.N))
        dataset = tf.data.Dataset.from_generator(self.generate, output_types, output_shapes)
        return dataset
    

    def get_batch(self):
        if type(self.As) is list:
            path = random.choice(self.As)
            self.A = np.load(path)
            self.A = tf.convert_to_tensor(self.A)

        dataset = self.dataset
        dataset = dataset.repeat(-1)
        dataset = dataset.prefetch(-1)
        dataset = dataset.batch(self.batchsize)
        return dataset


    def reduce_batchsize(dataset,batchsize=1):
        return dataset.unbatch().batch(batchsize)

    
    def filter_distance(_, s, distance=0.):
        x,y,_ = rg.pos()
        smap = PhyiscalModel.vector_to_sourcemap(s)
        idx = tf.where(tf.not_equal(smap,0.))
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        x = x[idx[0]]
        y = y[idx[1]]
        mask = tf.square(x) - tf.square(y) > tf.square(distance)
        out = tf.reduce_all(mask)
        return out

# %%