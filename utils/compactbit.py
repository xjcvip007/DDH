import numpy as np


def compactbit(b):
	'''
	b = bits array
	cb = compacted string of bits(using words of 'word' bits)
	'''
	b_mat = np.mat(b)
	[nSamples, nbits] = b_mat.shape
	nwords = int(np.ceil((float(nbits) / 8)))
	cb = np.zeros((nSamples,nwords),dtype = np.uint8)

	for i in xrange(nSamples):
		for j in xrange(nwords):
			temp = b[i , j * 8 : (j + 1) * 8]
			value = convert(temp)
			cb[i,j] = value

	return cb

def convert(arr):
	arr_mat = np.mat(arr)
	[_, col] = arr_mat.shape
	value = 0
	for i in xrange(col):
		value = value + (2 ** i) * arr[i]
	
	return value


if __name__ == '__main__':
	b = np.array([[0,0,1,1,0,0,0,1,1],[0,0,0,1,0,0,1,1,1]])
	print compactbit(b)
