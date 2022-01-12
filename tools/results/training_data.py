#%%
from tools.training.data import get_bg_batch
from tools.training.loss_funcs import nmse_db
from tools.model import A
from tensorflow import einsum, reduce_mean, convert_to_tensor
from matplotlib import pyplot as plt 
from numpy import log10, abs, load

train_data= get_bg_batch(A,200,SNR=40)
y,x = next(iter(train_data))

y_ = einsum("ij,kj->ki",A,x)

NMSE = (nmse_db(y,y_).numpy())
mean_x = reduce_mean(x[x!=0])
mean_y = reduce_mean(y)

signal = reduce_mean(abs(y)).numpy()
noise = reduce_mean(abs(y_ - y)).numpy()
ratio = signal / noise
snr = 10*log10(ratio)

print("NMSE   "+"{:.3}".format(NMSE))
print("mean_x "+"{:.2}".format(mean_x))
print("mean_y "+"{:.2}".format(mean_y))
print("SNR    "+"{:.3}".format(snr))
#%%

y  = y.numpy()[-1]
y_ = y_.numpy()[-1]

fig = plt.figure(figsize=(16,9),dpi=100)
ax  = fig.add_subplot(211)
ax.plot(y,"k",label="y")
ax.plot(y-y_,"r--",label="err")
ax.set_title("Simulated Vector with Error")

ax1 = fig.add_subplot(212)
ax1.plot(y_,"g",label="y_")
ax1.set_title("Calculated Vector")

ax.legend()
fig.show()
# %%
