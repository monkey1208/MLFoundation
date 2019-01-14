import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1208)
def get_noise_index(size=20,ratio=0.2):
	noise_indx = np.arange(size)
	np.random.shuffle(noise_indx)
	noise_indx = noise_indx[:int(size*ratio)]
	return noise_indx
	
size = 20
avg_Ein = []
avg_Eout = []
for times in range(1000):
	data = np.random.uniform(-1,1,size)
	noise_indx = get_noise_index()
	label = np.sign(data)
	newlabel = label.copy()
	np.negative.at(newlabel, noise_indx) # flip noise data
	# sort data & label
	sorted_indx = data.argsort()
	newlabel = newlabel[sorted_indx]
	data = data[sorted_indx]
	# get the best H
	min_Ein, best_s, best_theta = 1, 1, 1
	for s in [1,-1]:
		for i in range(size+1):
			# the first and last interval
			if i == 0:
				theta = data[0] - 1
			elif i == size:
				theta = data[-1] + 1
			else:
				theta = (data[i] + data[i-1])/2
			err = 0
			for x,y in zip(data, newlabel):
				if s*np.sign(x-theta) != y:
					err += 1
			err /= size
			if err <= min_Ein:
				min_Ein = err
				best_s = s
				best_theta = theta
	print('Ein = {}, s = {}, theta = {}'.format(min_Ein, best_s, best_theta))
	Eout = 0.5 + 0.3*best_s*(np.absolute(best_theta)-1)
	avg_Ein.append(min_Ein)
	avg_Eout.append(Eout)
print('avg_Ein = {}'.format(np.average(avg_Ein)))
print('avg_Eout = {}'.format(np.average(avg_Eout)))
avg_Ein = np.array(avg_Ein)
avg_Eout = np.array(avg_Eout)
avg_diff = avg_Ein-avg_Eout
plt.hist(avg_diff,bins=int((max(avg_diff)-min(avg_diff))/0.01))#, bins=range(min(avg_diff), max(avg_diff),0.2))
plt.xlabel('Ein-Eout')
plt.ylabel('Frequency')
plt.savefig('hw2_7.png')

