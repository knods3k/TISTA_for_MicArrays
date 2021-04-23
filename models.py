import tensorflow as tf
from tensorflow import keras
from helper import simple_soft_threshold

eta = simple_soft_threshold

class TISTALayer(keras.layers.Layer):
	def __init__(self,T,A,W,initial_lambda,initial_gamma):
		super(TISTALayer, self).__init__()
		self._name = "TISTA_" + str(T)

		self.A = A
		self.W = W

		self.lam = tf.Variable(initial_lambda,name="lam"+str(T),trainable=True)
		self.gam = tf.Variable(initial_gamma,name="gam"+str(T),trainable=True)
		
	def call(self, s,y):
		s = eta(s + self.gam * tf.einsum("ij,kj->ki",self.W,y - tf.einsum("ij,kj->ki",self.A,s)),self.lam)
		return s

class TISTA(keras.Model):
	def __init__(self,A,W,initial_lambda=0.0,initial_gamma=1.0,T=6):
		super(TISTA, self).__init__()
		self.T = T
		
		self.A 		= A
		self.W		= W	
		self.lyrs 	= []
		for t in range(T):
			self.lyrs.append(TISTALayer(t,self.A,self.W,initial_lambda,initial_gamma))

	def call(self,y,training=False):
		s = tf.matmul(y,self.A)*0.0
		for layer in self.lyrs:
			s = layer(s,y)
		return s

# LISTA NOT WORKING YET

class LISTALayer(keras.layers.Layer):
	def __init__(self,S,initial_lambda):
		super(LISTALayer, self).__init__()
		self.S_ = S
		self.lam = tf.Variable(initial_lambda)
	def call(self, xhat_,By_):
		# xhat_ = eta( tf.einsum("ij,kl ->  l" ,self.S_,xhat_) + By_, self.lam )
		xhat_ = eta( tf.einsum("ij,kl ->  l" ,self.S_,xhat_),self.lam)
		# xhat_ = tf.ones((5202))
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
		
		B = A.T / (1.01 * tf.linalg.norm(A,2)**2)
		self.B_ =  tf.Variable(B,dtype=tf.float32,name='B_0',trainable=True)

		initial_lambda = initial_lambda*tf.ones( (N,1),dtype=tf.float32 ) # this might not be the case ( depends if iid signal)
		self.lam0_ = tf.Variable( initial_lambda,name='lam',trainable=True)

		S = tf.eye(N) - tf.matmul(B,A)
		self.S_ = tf.Variable(S,dtype=tf.float32,name='S_0',trainable=True)
		
		self.lyrs = []
		for _ in range(T):
			self.lyrs.append(LISTALayer(self.S_,initial_lambda))

	def call(self,y, training=False):
		#inputs = keras.Input(shape=(250,), name="csm")
		By_ = tf.matmul( self.B_ , y )
		xhat_ = eta(By_, self.lam0_)

		for layer in self.lyrs:
			xhat_ = layer(xhat_,By_)
		return xhat_
