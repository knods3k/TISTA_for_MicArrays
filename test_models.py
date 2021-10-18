#%%
import tensorflow as tf
from numpy import mean, load, count_nonzero
import pandas as pd
from bernoulli_gaussian import get_bg_batch
from reader import get_batch
from matplotlib import pyplot as plt


def MSE(tup):
	x = tup[0]
	y = tup[1]
	diff = x - y
	square = diff**2
	mse = mean(square)
	return mse

def reshape_sourcemap(x):
	N = x.shape[0]
	gridsize = N//2
	gridlen	 = int(gridsize**0.5)

	x = x[:gridsize].reshape(gridlen,gridlen)
	return x

TESTSIZE = 10
BATCHSIZE = 1

SNR = 40
HE = 16
WORST = 0

FREQ = 343 * HE
MODEL = f"{FREQ}Hz_{SNR}SNR"
SAVEFILE = f"He={HE}_SNR={SNR}_"
if not WORST:
	SAVEFILE += "bestcase"
else:
	SAVEFILE += "worstcase"
A = load(f"data/A_{FREQ}Hz.npy")
model = tf.keras.models.load_model("models/"+MODEL)
train_data = get_bg_batch(A,BATCHSIZE,SNR=SNR,noise=False).repeat()


if __name__ == "__main__":
	out = []
	for y,x in train_data.take(TESTSIZE):
		pred = model.predict(y)
		true = x.numpy()
		out.append([pred,true])

	data = pd.DataFrame(out,columns=["Pred","True"])

	data["MSE"] 	= [MSE(tup) for tup in tuple(zip(data["Pred"],data["True"]))]
	data["Nonzero"] = data["True"].map(count_nonzero)
	data = data.sort_values("Nonzero")

	sourcemaps = []
	for idx in [0,1,2]:
		if WORST:
			idx -= -3
		row = data.iloc[idx]
		pred = row["Pred"]
		true = row["True"]
		mse	 = row["MSE"]
		mse = "{:.0e}".format(mse)

		pred = reshape_sourcemap(pred[0])
		true = reshape_sourcemap(true[0])

		sourcemaps.append([pred,true,mse])


	nplots = len(sourcemaps)
	fig, ax = plt.subplots(2,nplots,sharex=True,sharey=True)
	ax[0,0].set_ylabel("True")
	ax[1,0].set_ylabel("Predicted")

	for i,map in enumerate(sourcemaps):
		pred = map[0]
		true = map[1]
		mse  = map[2]

		ax[0,i].imshow(true)
		ax[1,i].set_xlabel("MSE = "+mse)
		ax[1,i].imshow(pred)

	fig.size = (32,24)
	# plt.savefig("plots/"+SAVEFILE+".png",dpi=100)
	fig.show()

#%%
	for idx in [TESTSIZE-(1+i) for i in [0,1,2]]:
		row = data.loc[idx]
		pred = row["Pred"][0]
		true = row["True"][0]
		mse	 = row["MSE"]
		mse = "{:.0e}".format(mse)
		plot_sourcemaps(pred,true,title=mse)



# %%

