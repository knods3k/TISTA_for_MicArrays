#%%
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (16*.8,9*.8)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 14
plt.rc("image",cmap="hot_r")

if __name__ == "__main__":
	img = np.zeros((26,26))
	fig, ax = plt.subplots(1,3,sharex=True,sharey=True)


	img = np.random.rand(26*26).reshape((26,26))
	img1 = img
	ax[0].imshow(img1,extent=[-.5,.5,-.5,.5])
	ax[0].set_title(r"$r_1$")
	ax[0].set_xlabel("x")
	ax[0].set_ylabel("y")

	img = np.random.rand(26*26).reshape((26,26))
	img = img < 0.01
	img = img.astype(float)
	img2 = img
	ax[1].imshow(img2,extent=[-.5,.5,-.5,.5])
	ax[1].set_title(r"$\mathcal{H}(r_1,q)$")
	ax[1].set_xlabel("x")

	normal = np.random.normal(0,1,(26,26))
	normal = np.abs(normal)
	img = img * normal
	img3 = img
	ax[2].imshow(img3,extent=[-.5,.5,-.5,.5])
	ax[2].set_title(r"$\mathcal{H}(r_1,q)|r_2|$")
	ax[2].set_xlabel("x")

# %%
fig.savefig("data/plots/data_generation.pdf")
# %%
