import readfile
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
def sign(s):
	return -1 if s<=0 else 1

def PLA(X, Y):
	w = np.zeros(5)
	shuffle_indices = np.arange(X.shape[0])
	np.random.shuffle(shuffle_indices)
	X, Y = X[shuffle_indices], Y[shuffle_indices]
	steps, updates, correct_steps = 0, 0, 0
	while(correct_steps < X.shape[0]):
		out = np.dot(X[steps], w) # <=0 wrong  >0 correct
		if sign(out) != Y[steps]: #wrong
			w += Y[steps]*X[steps]
			correct_steps = 0
			updates += 1
		else: #correct
			correct_steps += 1
		steps = (steps + 1) % X.shape[0]
	return w, updates

def main():
	fname = sys.argv[1]
	X, Y = readfile.readfile(fname)
	X, Y = np.array(X), np.array(Y)
	times = []
	np.random.seed(1208)
	for i in range(1126):
		w, updates = PLA(X, Y)
		times.append(updates)
		print(w, updates)
	print('average update steps : {}'.format(np.average(times)))
	plt.hist(np.array(times), bins=range(min(times), max(times),1))
	plt.xlabel('Update counts')
	plt.ylabel('Counts')
	plt.savefig('hw1_7.png')

if __name__ == '__main__':
	main()

