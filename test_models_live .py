#%%
from test_models import *
from main import train_data
import cv2 as cv


TESTSIZE = 10
BATCHSIZE = 1
MODEL = "He15.0"
TRAINING    = "data/quicktest.tfrecord"

model = tf.keras.models.load_model("models/"+MODEL)
train_data = get_batch(TRAINING,BATCHSIZE).repeat()
train_data = get_bg_batch(A,BATCHSIZE,SNR=10,noise=True).repeat()

if __name__ == "__main__":
	win = cv.namedWindow("image",flags=cv.WINDOW_KEEPRATIO)
	for y,x in train_data.take(TESTSIZE):
		pred = model.predict(y)
		true = x.numpy()

		pred = reshape_sourcemap(pred)
		true = reshape_sourcemap(true)

		img = true
		img = img/np.max(img)
		cv.imshow("image", img)
		cv.waitKey(1000)
		img = pred
		img = img/np.max(img)
		cv.imshow("image", img)
		cv.waitKey(1000)
	cv.destroyAllWindows()

# %%
