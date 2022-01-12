import tensorflow as tf
from tensorflow import keras
from numpy import pi

def simple_soft_threshold(r, lam):
	# lam = tf.maximum(lam, 0)
	lam = tf.nn.relu(lam)
	return tf.sign(r) * tf.maximum(tf.abs(r) - lam, 0)

def mmse(y,sig,a,p):
	sig = tf.maximum(sig,0)
	a = tf.maximum(a,0)
	xi = sig + a
	F = lambda z,v: (2*pi*v)**-0.5 * tf.exp((-z**2) / (2*v))
	return ((y*a)/xi) * (p*F(y,xi)) / ((1-p)*F(y,sig) + (p)*F(y,xi))


eta = simple_soft_threshold

class TISTA(keras.Model):
	def __init__(self,A,initial_lambda=0.1,initial_gamma=1.0,T=6):
		super(TISTA, self).__init__()
		self.T = T
		self.T_save = tf.Variable(self.T,name="T",trainable=False)

		self.A 	= A
		self.W	= tf.linalg.pinv(A)
		self.A_save = tf.Variable(self.A,name="A",trainable=False)
		self.W_save = tf.Variable(self.W,name="W",trainable=False)
		self.lams 	= []
		self.gams 	= []
		for t in range(T):
			self.lams.append(tf.Variable(initial_lambda,name="lam"+str(t),trainable=True))
			self.gams.append(tf.Variable(initial_gamma,name="gam"+str(t),trainable=True))
	
	def call(self,y,training=True,T=None):
		s = tf.matmul(y*0.0,self.A)
		for lam,gam in zip(self.lams[:T],self.gams[:T]):
			s = eta(s + gam * tf.einsum("ij,kj->ki",self.W,y - tf.einsum("ij,kj->ki",self.A,s)),lam)
		return  tf.nn.relu(s)

class TISTA_mmse(keras.Model):
	def __init__(self,A,initial_lambda=0.1,initial_gamma=1.0,T=6):
		super(TISTA_mmse, self).__init__()
		self.T = T

		self.a 		= tf.Variable(1.0,name="a",trainable=True)
		self.p		= tf.Variable(0.1,name="p",trainable=True)

		self.A 		= A
		self.W		= W = tf.linalg.pinv(A)
		self.lams 	= []
		self.gams 	= []
		for t in range(T):
			self.lams.append(tf.Variable(initial_lambda,name="lam"+str(t),trainable=True))
			self.gams.append(tf.Variable(initial_gamma,name="gam"+str(t),trainable=True))

	def call(self,y,training=True):
		s = tf.matmul(y*0.0,self.A)
		for lam,gam in zip(self.lams,self.gams):
			s = mmse(s + gam * tf.einsum("ij,kj->ki",self.W,y - tf.einsum("ij,kj->ki",self.A,s)),lam,self.a,self.p)
		return s #tf.nn.relu(s)

class LISTA(keras.Model):
	def __init__(self,A,initial_lambda=0.0,T=6):
		super(LISTA, self).__init__()
		self.T = T

		self.A = A
		M,N = A.shape
		
		B = tf.transpose(A) / (1.01 * tf.linalg.norm(A,2)**2)
		self.B =  tf.Variable(B,dtype=tf.float32,name='B',trainable=True)

		self.lams 	= []
		for t in range(T):
			self.lams.append(tf.Variable(initial_lambda,name="lam"+str(t),trainable=True))
		
		S = tf.eye(N) - tf.matmul(B,A)
		self.S = tf.Variable(S,dtype=tf.float32,name='S',trainable=True)
		
	def call(self,y, training=True):
		s = tf.matmul(y*0.0,self.A)
		for lam in self.lams:
			s = eta(tf.einsum("ij,kj->ki",self.S,s) + tf.einsum("ij,kj->ki",self.B,y),lam)
		return s #tf.nn.relu(s)