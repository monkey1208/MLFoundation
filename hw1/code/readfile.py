def readfile(fname):
	X = []
	Y = []
	with open(fname, 'r') as f:
		for line in f:
			datas = line.replace('\t',' ').strip('\n').split(' ')
			x = [1]*5
			x[1:] = datas[:4]
			X.append([float(i) for i in x])
			Y.append(int(datas[-1]))
	return X, Y
