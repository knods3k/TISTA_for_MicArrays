#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import os

plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams["figure.dpi"] = 100

MODEL 	= "1372"
PATH 	= os.path.join("models/",MODEL)

# try:
# 	model_weights

# except:
# 	model_weights = []
# 	for dir in os.listdir("models/"):
# 		if "He15" in dir:
# 			model = tf.keras.models.load_model(os.path.join("models/",dir))
# 			ws = [w.numpy() for w in model.weights]
# 			lams = ws[:len(ws)//2]
# 			gams = ws[len(ws)//2:]
# 			model_weights.append([dir,lams,gams])

# SNR20 = [name for name in model_weights if "_20SNR" in name[0]]
# SNR40 = [name for name in model_weights if "_40SNR" in name[0]]
# SNR60 = [name for name in model_weights if "_60SNR" in name[0]]

# f1372 = [name for name in model_weights if "1372" in name[0]]
# f2744 = [name for name in model_weights if "2744" in name[0]]
# f5488 = [name for name in model_weights if "5488" in name[0]]


# model = tf.keras.models.load_model(os.path.join("models/","He15.0_sim"))
# ws = [w.numpy() for w in model.weights]
# lams = ws[:len(ws)//2]
# gams = ws[len(ws)//2:]
# model_weights.append([dir,lams,gams])

model = tf.keras.models.load_model(PATH)

model_weights = []
ws = [w.numpy() for w in model.weights]
lams = ws[:len(ws)//2]
gams = ws[len(ws)//2:]
model_weights.append([MODEL,lams,gams])


fig = plt.figure()
for model in model_weights:
	name = model[0]
	lams = model[1]
	gams = model[2]
	
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	ax1.set_title("Lambdas")
	ax1.plot(lams,label=name)
	ax2.set_title("Gammas")
	ax2.plot(gams,label=name)

ax1.legend()
fig.show()



# %%
