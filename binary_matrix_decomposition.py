import numpy as np
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
# import seaborn as sns
# sns.set_style('whitegrid')



def bin_random_mat(m,n,p_0 = 0.5):
	"""
	return a 0-1 valued random matrix
	"""
	return np.array((np.random.randn(m,n) >= p_0), dtype = np.float)

def Metropolis_transition(X, p_0 = 0.5, shake_times = 5):
	"""
	symetric metropolis kernel for 0-1 valued matrix
	"""
	p,q = np.shape(X)
	X_iter = X
	for i in range(shake_times):
		m,n = np.random.choice(range(p)),np.random.choice(range(q))
		
		X_iter[m,n] = np.float(X_iter[m,n] == 0)
	return X_iter
# test for Metropolis kernel
# A = bin_random_mat(3,2)
# print (A)
# print (Metropolis_transition(A))	



# def norm_matrix(A):
# 	"""
# 	Frobenius norm for test.
# 	"""
# 	return np.linalg.norm(A)


def norm_matrix(A):
	"""
	norm of matrix for test
	"""
	X = np.abs(A)
	return np.sum([np.sum(X[i]) for i in range(np.shape(X)[0])])



def V_potential(X,A):
	"""
	potential function to minimise
	"""
	return norm_matrix(A-X)





def binary_dec(A,n_iter = 1000):
	"""
	The function return the decomposition of matrix in finite field 2Z/Z, i.e. 0-1 matrix.
	The shape of matrix A: p * q
	We want to decompose A into B * C, 
	where B is of shape p * p;
		  C is of shape p * q;
	and B * B^t = Id

	"""

	### Initialization ###

	p, q = np.shape(A)
	### B : to be changed
	B = np.eye(p)
 	###
	C = bin_random_mat(p,q)
	list_dist = []
	B_argmin = B
	C_argmin = C




	## temperature ##
	T_n = np.log(np.arange(2,n_iter+2,1))
	#T_n = np.arange(2,n_iter+2,1)
	for i in range(n_iter):
	## update ##
		C_0 = np.matrix(C)
		list_dist =np.append( list_dist, V_potential(np.dot(B,C_0),A) )
		if V_potential(np.dot(B_argmin,C_argmin),A) == 0:
			break
	########## transition #############
	# Here we take 2 steps independent(for B and for C respectively)
	# We could alse use metropolis hasting kernel.

		C_iter = np.matrix(Metropolis_transition(C))
	

		B_iter = B[np.random.permutation(np.arange(p))]
		
		if np.random.uniform(0,1,1) < \
				np.exp(-1./T_n[i]*( V_potential(np.dot(B_iter,C_iter), A)\
				 - V_potential(np.dot(B,C_0),A)  ) ):
			C = C_iter
			#B = B_iter
	######### end of transition ##############

		if V_potential(np.dot(B,C),A) < np.min(list_dist):
			
			B_argmin = B
			C_argmin = np.matrix(C)
			# print i+1
			# print V_potential(np.dot(B_argmin,C_argmin),A)
			# print C_argmin
			# print '\n'


		
		
	

	return list_dist,B_argmin, C_argmin








############## test #############
### size of matrix A
p = 5
q = 5
print ('number of states : %s' % 2**(p*q))

A = bin_random_mat(p,q)

print ('calculating...')

l, B_1, C_1 = binary_dec(A = A,n_iter = 100000)

#print l
print ('min: %s' % np.min(l))
print('argmin:%s'%np.argmin(l))

############### plot ################

# plt.figure(figsize = [15,5])
# plt.plot(l,color = 'grey',label = 'potential_trace')
# minimum, = plt.plot(np.argmin(l),np.min(l),'ro',label = 'minimum', ls = '')
# plt.legend(handler_map={ minimum : HandlerLine2D(numpoints=1)})
# plt.title('Illustration of the trace of potential')
# plt.show()

############### comparason $ status ################
print("A: ")
print(A)
print("B: ")
print(B_1)
print("C: ")
print(C_1)
print("B*C: ")
print(np.dot(B_1,C_1))
print('difference (A - B*C) : ')
print (A - np.dot(B_1,C_1))
print ("distance: ")
print V_potential(np.dot(B_1,C_1),A)
print ('infomation lost: %s' % (V_potential(np.dot(B_1,C_1),A)/p/q))



