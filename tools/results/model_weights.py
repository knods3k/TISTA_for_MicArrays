#%%
from os.path import normpath, join
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams["figure.dpi"] = 100
plt.rc("font",size=14)

BASEPATH = normpath(join("models","Clean"))

HE = 16
T = 40
MODELNAME = f"He={HE}_T={T}"
PATH = normpath(join(f"{BASEPATH}",f"{MODELNAME}"))


model = tf.keras.models.load_model(PATH)
#%%
model_weights = []
T = model.T_save.numpy()
ws = [w.numpy() for w in model.trainable_weights]
lams = ws[:T]
gams = ws[T:]
model_weights.append([MODELNAME,lams,gams])




#%%


fig = plt.figure()
for model in model_weights:
	name = model[0]
	lams = model[1]
	gams = model[2]
	
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	ax1.plot(lams)
	ax1.set_ylabel(r"Meshsize $\lambda$")
	ax2.plot(gams)
	ax2.set_ylabel(r"Stepsize $\beta$")
	ax2.set_xlabel(r"Layer $T$")
	for ax in [ax1, ax2]:
		#ax.set_xticks(range(1,T+1,3))
		pass

#ax1.legend()
fig.show()



# %%
fig.savefig("data/plots/model_weights.pdf")
# %%
