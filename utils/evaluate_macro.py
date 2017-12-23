import numpy as np


def evaluate_macro(Rel, Ret):
	'''
	evaluate macro_averaged performance
	Input:
		Rel = relevant  train documents for each test document
		Ret = retrieved train documents for each test document

	Output:
		p   = macro_averaged precision
		r   = macro_averaged recall  
	'''
	Rel_mat = np.mat(Rel)
	numTest = Rel_mat.shape[1]
	print 'numTest=',numTest
	precisions = np.zeros((numTest))
	recalls    = np.zeros((numTest))

	retrieved_relevant_pairs = (Rel & Ret)

	for j in xrange(numTest):
		retrieved_relevant_num = len(retrieved_relevant_pairs[:,j][np.nonzero(retrieved_relevant_pairs[:,j])])
		#print 'retrieved_relevant_num=',retrieved_relevant_num
		retrieved_num = len(Ret[:, j][np.nonzero(Ret[:, j])])
		#print 'retrieved_num=',retrieved_num
		relevant_num  = len(Rel[:, j][np.nonzero(Rel[:, j])])
		#print 'relevant_num=',relevant_num
		
		if retrieved_num:
			#print 1
			precisions[j] = float(retrieved_relevant_num) / retrieved_num
		
		else:
			precisions[j] = 0.0

		if relevant_num:
			recalls[j]    = float(retrieved_relevant_num) / relevant_num
		
		else:
			recalls[j]    = 0.0

	p = np.mean(precisions)
	r = np.mean(recalls)
	return p,r


if __name__ == '__main__':
	cateTrainTest = np.array([[True, True, False, False],[False, False, False, True]])
	Ret = np.array([[False, True, False, True],[True, True, False, False]])
	evaluate_macro(cateTrainTest, Ret)