import numpy as np


def cat_apcal(traingnd, testgnd, IX, top_N):
	'''
	ap = apcal(score, label)
	average precision (AP) calculation
	'''

	[numtrain, numtest] = IX.shape

	apall = np.zeros(numtest)

	for i in xrange(numtest):
		y = IX[:, i]
		x = 0
		p = 0
		new_label = np.zeros((1,numtrain))
		new_label[traingnd.T == testgnd[i]] = 1
		num_retuen_NN = numtrain
		for j in xrange(num_retuen_NN):
			if new_label[0, y[j]] == 1:
				x = x + 1
				p = p + float(x) / (j + 1)
		if p == 0:
			apall[i] = 0
		else:
			apall[i] = p / x

	pall = np.zeros(numtest)
	for ii in xrange(numtest):
		y_1 = IX[:,ii]
		n = 0
		new_label_1 = np.zeros((1,numtrain))
		new_label_1[traingnd.T == testgnd[ii]] = 1
		for jj in xrange(top_N):
			if new_label_1[0,y_1[jj]] == 1:
				n = n + 1
		pall[ii] = 1.0 * n / top_N
		# print ii,pall[ii]

	ap = np.mean(apall)
	p_topN = np.mean(pall)
	return ap,p_topN


if __name__ == '__main__':
	traingnd = np.array([[1],[4],[2],[1]])
	testgnd  = np.array([[2],[4]])
	IX       = np.array([[1,2],[2,1],[3,1],[0,2]])
	cat_apcal(traingnd, testgnd, IX)
