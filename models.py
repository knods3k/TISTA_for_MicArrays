import tensorflow as tf
from tensorflow import keras
from helper import simple_soft_threshold

eta = simple_soft_threshold

class TISTALayer(keras.layers.Layer):
	def __init__(self,Wt,initial_lambda,initial_gamma):
		super(TISTALayer, self).__init__()
		self.Wt_ = Wt
		self.lam = tf.Variable(initial_lambda,name="lam",trainable=True)
		self.gam = tf.Variable(initial_gamma,name="gam",trainable=True)
	def call(self, xhat_,yWt_):
		r = xhat_ + self.gam * yWt_
		xhat_ = eta(r,self.lam)
		return xhat_

class TISTA(keras.Model):
	def __init__(self,W,initial_lambda=0.1,initial_gamma=1.0,T=6):
		super(TISTA, self).__init__()
		self.T = T
		
		Wt = tf.transpose(W)

		self.Wt_ = tf.Variable(Wt,dtype=tf.float32,name='Wt_',trainable=False)
		self.lyrs = []
		for t in range(T):
			self.lyrs.append(TISTALayer(Wt,initial_lambda,initial_gamma))

	def call(self,y,training=False):
		yWt_ = tf.matmul(y,self.Wt_)
		xhat_ = 0.0
		for layer in self.lyrs:
			xhat_ = layer(xhat_,yWt_)
		return xhat_

# LISTA NOT WORKING YET

class LISTALayer(keras.layers.Layer):
	def __init__(self,S,initial_lambda):
		super(LISTALayer, self).__init__()
		self.S_ = S
		self.lam = tf.Variable(initial_lambda)
	def call(self, xhat_,By_):
		xhat_ = eta( tf.matmul(self.S_,xhat_) + By_, self.lam )
		return xhat_

class LISTA(keras.Model):
	"""
	Trainable Parameters across all layers for A (250,500): 
		S: shape (N,N)    -> (500 x 500) = 250000 parameters
		B: shape (N,M)    -> (500,250) = 125000 parameters
		lambda: (N)        -> 500 = 500*T+1 -> für T=6 -> 3500
		results in 378,500 parameters

		S should be the same across layers:
			check S for different layers after training: model.layers[1].get_weights()[0].shape
	"""
	def __init__(self,A,initial_lambda=0.1,T=6):
		super(LISTA, self).__init__()
		self.T = T
		M,N = A.shape
		initial_lambda = initial_lambda*tf.ones( (N,1),dtype=tf.float32 ) # this might not be the case ( depends if iid signal)
		B = A.T / (1.01 * tf.linalg.norm(A,2)**2)
		self.B_ =  tf.Variable(B,dtype=tf.float32,name='B_0',trainable=True)
		self.lam0_ = tf.Variable( initial_lambda,name='lam',trainable=True)
		S_ = tf.Variable( tf.eye(N) - tf.matmul(B,A),dtype=tf.float32,name='S_0',trainable=True)
		# depends on the number of layes
		self.lyrs = []
		for _ in range(T):
			self.lyrs.append(LISTALayer(S_,initial_lambda))

	def call(self,y, training=False):
		#inputs = keras.Input(shape=(250,), name="csm")
		By_ = tf.matmul( self.B_ , y )
		xhat_ = eta(By_, self.lam0_)
		for layer in self.lyrs:
			xhat_ = layer(xhat_,By_)
		return xhat_
