#%%
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi
import tensorflow as tf

def mmse(y,sig,a,p):
	sig = sig**2
	a = a**2
	xi = sig + a
	F = lambda z,v: (2*pi*v)**-0.5 * tf.exp((-z**2) / (2*v))
	return ((y*a)/xi) * (p*F(y,xi)) / ((1-p)*F(y,sig) + (p)*F(y,xi))

x = np.linspace(-10,10,100)
y = mmse(x,0.2,1.,0.1)

fig,ax = plt.subplots(1)
ax.plot(x,y,"k")
ax.plot(x,x,"k--")
# %%
