import matplotlib.pyplot as plt
import sys
import numpy as np
import math

def calculate_grad(y, w, x):
	return (-y*x)*sigmoid(-y*(w.dot(x)))

def sigmoid(s):
	return  1/(math.exp(-s)+1)

def read_data(fname):
	X = []
	Y = np.array([])
	with open(fname, 'r') as f:
		for line in f:
			data = line.strip().split(' ')
			x = np.array([1]+data[:-1]).astype(np.float)
			y = np.array(data[-1]).astype(np.int)
			X.append(x)
			Y = np.append(Y, y)
	X = np.array(X)
	return X, Y

def gradient_descent(X, Y, X_test, Y_test, lr=0.01, It=2000):
	W = np.zeros(X.shape[1])
	errors = []
	for i in range(It):
		gradients = np.zeros(X.shape[1])
		for (x, y) in zip(X, Y):
			gradient = calculate_grad(y, W, x)
			gradients += gradient
		gradients /= X.shape[0]
		#gradient descent
		W -= gradients*lr
		y_pred = X_test.dot(W)
		err = (np.sum(np.sign(y_pred)*Y_test < 0))/X_test.shape[0]
		errors.append(err)
	return errors

def stochastic_gradient_descent(X, Y, X_test, Y_test, lr=0.001, It=2000):
	W = np.zeros(X.shape[1])
	errors = []
	for i in range(It):
		x, y = X[i % X.shape[0]], Y[i % X.shape[0]]
		gradient = calculate_grad(y, W, x)
		#gradient descent
		W -= gradient*lr
		y_pred = X_test.dot(W)
		err = (np.sum(np.sign(y_pred)*Y_test < 0))/X_test.shape[0]
		errors.append(err)
	return errors

def main():
	train = sys.argv[1]
	test = sys.argv[2]
	X, Y = read_data(train)
	X_test, Y_test = read_data(test)
	It = 2000
	errors = gradient_descent(X, Y, X_test, Y_test, It=It)
	errors2 = stochastic_gradient_descent(X, Y, X_test, Y_test, It=It)
	l1, = plt.plot(range(1, It+1), errors, label='gradient descent')
	l2, = plt.plot(range(1, It+1), errors2, label='stochastic gradient descent')
	plt.legend(loc='best')
	plt.xlabel('t')
	plt.ylabel('error')
	plt.title('Eout')
	plt.savefig('hw3_5.png')
	#plt.show()

if __name__ == '__main__':
	main()
