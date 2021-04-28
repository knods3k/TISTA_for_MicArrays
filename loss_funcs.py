import tensorflow as tf 

def simple_soft_threshold(r_, lam_):
    "implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)"
    lam_ = tf.maximum(lam_, 0)
    return tf.sign(r_) * tf.maximum(tf.abs(r_) - lam_, 0)

def log10(x):
	'''
	at the moment no log10 in tensorflow implemented..
	'''
	numerator = tf.math.log(x)
	denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
	return numerator / denominator

def mse(x_true, x_pred):
	squared_difference = tf.square(x_pred-x_true)
	return tf.reduce_mean(squared_difference)

def sse(x_true, x_pred):
	squared_difference = tf.square(x_pred-x_true)
	return tf.reduce_sum(squared_difference)

def nmse_db(x_true, x_pred):
	nmse_denom_ = tf.reduce_mean(tf.square(x_true))
	squared_difference = tf.square(x_pred-x_true)
	return 10*log10(tf.reduce_mean(squared_difference)/ nmse_denom_)

def nmse(x_true, x_pred):
	nmse_denom_ = tf.reduce_mean(tf.square(x_true))
	squared_difference = tf.square(x_pred-x_true)
	return tf.reduce_mean(squared_difference)/ nmse_denom_

def nsse(x_true, x_pred):
	nsse_denom_ = tf.reduce_mean(tf.square(x_true))
	squared_difference = tf.square(x_pred-x_true)
	return tf.reduce_sum(squared_difference)/ nsse_denom_
