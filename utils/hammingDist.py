import numpy as np


def hammingDist(B1, B2):
	'''
	Compute hamming distance between two sets of samples (B1, B2)
 	Dh=hammingDist(B1, B2);

 	Input
    	B1, B2: compact bit vectors. Each datapoint is one row.
    	size(B1) = [ndatapoints1, nwords]
    	size(B2) = [ndatapoints2, nwords]
    	It is faster if ndatapoints1 < ndatapoints2
 
 	Output
    	Dh = hamming distance. 
    	size(Dh) = [ndatapoints1, ndatapoints2]

 	example query
 	Dhamm = hammingDist(B2, B1);
 	this will give the same result than:
    Dhamm = distMat(U2>0, U1>0).^2;
	the size of the distance matrix is:
 	size(Dhamm) = [Ntest x Ntraining]
	'''
	#look-up table:
	bit_in_char = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 
    3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 
    3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 
    2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 
    3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 
    5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 
    2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 
    4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 
    4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 
    5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 
    5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8], dtype = np.uint16)


	n1 = B1.shape[0]
	n2, nwords = B2.shape

	Dh = np.zeros((n1, n2), dtype = np.uint16)
	for i in range(n1):
		for j in range(nwords):
			y = (B1[i, j] ^ B2[:, j]).T
			Dh[i, :] = Dh[i, :] + bit_in_char[y]
	return Dh

if __name__ == '__main__':
	B1 = np.array([[1,2,3],[4,3,2]])
	B2 = np.array([[3,1,3],[3,2,1],[4,2,1]])
	Dh = hammingDist(B1, B2)
	print (Dh)