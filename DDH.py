# -*- coding=utf-8 -*- #
#####################################################################################################################
# File Name : DDQH.py
# Author : Jie Lin
# mail : lj_jackie@163.com
# Created Time : Thu 27 Apr 2017 10:01:30 AM
# Description : The implementation for "Discriminative Deep Quantization Hashing for Scalable Face Image Retrieval"
#####################################################################################################################

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, merge, Input, BatchNormalization,\
	LocallyConnected2D, Lambda
from keras.optimizers import SGD, Adam, Adadelta, Nadam
from keras.callbacks import EarlyStopping,LearningRateScheduler,ModelCheckpoint
from keras.utils import np_utils
from keras.regularizers import l2
from keras import backend as K
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
import os, pickle
import time
from numpy.matlib import repmat
from utils.compactbit import *
from utils.hammingDist import *
from utils.evaluate_macro import *
from utils.cat_apcal import *
import copy

# DATASET_NAME = 'facescrub'
DATASET_NAME = 'youtubeface'

if DATASET_NAME == 'youtubeface':
	TRAIN_SET_PATH = '/your/file/dir/to/youtubeface_train_set'
	TEST_SET_PATH = '/your/file/dir/to/youtubeface_test_set'
	NB_CLASSES = 1595
else:
    TRAIN_SET_PATH = '/your/file/dir/to/facescrub_train_set'
    TEST_SET_PATH = '/your/file/dir/to/facescrub_test_set'
    NB_CLASSES = 530

WEIGHTS_SAVE_PATH = 'you/file/dir/to/weights'
WEIGHTS_FILE_NAME = 'best/weights/dir/to/load'

NB_EPOCHS = 200
HASH_NUM = 48
SPLIT_NUM = 4
TOP_K = 50
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
LOSS_01_LAYER_PARAMS = 0.01
REGULARIZER_PARAMS = 0.0001

# True for train, False for predict
RUN_OR_CHECK = False
# RUN_OR_CHECK = True

def check_path_valid(path):
	return path if path.endswith('/') else path + '/'

def load_data_split_pickle(dataset): #you can change this method to load your dataset
	def get_files(vec_folder):
		file_names = os.listdir(vec_folder)
		file_names.sort()
		vec_folder = check_path_valid(vec_folder)
		for i in xrange(len(file_names)):
			file_names[i] = vec_folder + file_names[i]
		return file_names

	def load_data_xy(file_names):
		datas  = []
		labels = []
		for file_name in file_names:
			with open(file_name, 'rb') as f:
				x, y = pickle.load(f)
			datas.append(x)
			labels.append(y)
		data_array = np.vstack(datas)
		label_array = np.hstack(labels)
		return data_array, label_array

	test_folder, train_folder = dataset
	test_file_names = get_files(test_folder)
	train_file_names = get_files(train_folder)
	test_set = load_data_xy(test_file_names)
	train_set = load_data_xy(train_file_names)
	train_set_x, train_set_y = train_set[0], train_set[1]
	test_set_x, test_set_y = test_set[0], test_set[1]
	return [(train_set_x, train_set_y), (test_set_x, test_set_y)]

def deprocess_image(x):
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1
	x = np.clip(x, -1, 1)
	return x

def loss_01(x):
	return K.mean(K.abs(K.abs(x) - 1), axis=-1)

def loss_01_ouput_shape(input_shape):
	shape = list(input_shape)
	assert len(shape) == 2
	shape[-1] = 1
	return tuple(shape)

def split_output_shape(input_shape):
	shape = list(input_shape)
	assert len(shape) == 2
	shape[-1] = SPLIT_NUM
	return tuple(shape)

def face_hash_01_loss(y_true, y_pred):
	return y_pred

def face_hash_mean_loss(y_true, y_pred):
	return y_pred

def build_DDN_net(hash_num, split_num,REGULARIZER_PARAMS):
	main_input = Input(shape=(3, 32, 32), name='main_input')
	C1 = Convolution2D(20, 3, 3, border_mode='valid', init='he_uniform',
					   W_regularizer=l2(REGULARIZER_PARAMS))(main_input)
	B1 = BatchNormalization(axis=1)(C1)
	A1 = Activation('relu')(B1)
	M1 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(A1)
	C2 = Convolution2D(40, 2, 2, border_mode='valid', init='he_uniform',
					   W_regularizer=l2(REGULARIZER_PARAMS))(M1)
	B2 = BatchNormalization(axis=1)(C2)
	A2 = Activation('relu')(B2)
	M2 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(A2)
	C3 = Convolution2D(60, 2, 2, border_mode='valid', init='he_uniform',
					   W_regularizer=l2(REGULARIZER_PARAMS))(M2)
	B3 = BatchNormalization(axis=1)(C3)
	A3 = Activation('relu')(B3)
	M3 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(A3)
	M3_flatten = Flatten()(M3)
	C4 = LocallyConnected2D(80, 2, 2, border_mode='valid', init='he_uniform',
							W_regularizer=l2(REGULARIZER_PARAMS))(M3)
	B4 = BatchNormalization(axis=1)(C4)
	A4 = Activation('relu')(B4)
	C4_flatten = Flatten()(A4)
	Merge_layer = merge([M3_flatten, C4_flatten], mode='concat', concat_axis=1)
	Deepid_layer = Dense(hash_num * split_num, name='face_feature_layer', W_regularizer=l2(REGULARIZER_PARAMS))(
		Merge_layer)
	B5 = BatchNormalization()(Deepid_layer)
	A5 = Activation('relu')(B5)

	# DAE Modul in DDH
	outs = []
	for i in xrange(hash_num):
		slice_layer = Lambda(lambda x: x[:, split_num * i:split_num * i + split_num], output_shape=split_output_shape)(
			A5)
		fuse_layer = Dense(1, W_regularizer=l2(REGULARIZER_PARAMS))(slice_layer)
		outs.append(fuse_layer)
	Hash_layer = merge(outs, mode='concat', concat_axis=1, name='hash_layer')

	B6 = BatchNormalization(name='B6')(Hash_layer)
	A6 = Activation('tanh',name='A6')(B6)
	Softmax_layer = Dense(NB_CLASSES, activation='softmax', name='softmax_layer', W_regularizer=l2(REGULARIZER_PARAMS))(
		A6)
	loss_01_layer = Lambda(loss_01, output_shape=loss_01_ouput_shape, name='loss_01_layer')(A6)
	model = Model(input=main_input, output=[Softmax_layer, loss_01_layer])
	return model

def model_train(train_set_x, train_set_y):
	adam = Adam()
	earlystop = EarlyStopping(monitor='loss', patience=50, mode='min')
	mcp = ModelCheckpoint(WEIGHTS_SAVE_PATH + 'weights.{epoch:02d}-{loss:.3f}.hdf5', monitor='loss', verbose=0,
						  save_best_only=True, save_weights_only=True, mode='min')
	model = build_DDN_net(HASH_NUM, SPLIT_NUM, REGULARIZER_PARAMS)
	model.summary()
	print 'train start time: '+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
	print 'loss_01_layer_param is {0}'.format(LOSS_01_LAYER_PARAMS)
	model.compile(loss={'loss_01_layer':face_hash_01_loss, 'softmax_layer':'categorical_crossentropy'},
			   optimizer=adam,loss_weights={'loss_01_layer':LOSS_01_LAYER_PARAMS, 'softmax_layer':1.0}, metrics={'softmax_layer':'accuracy'})
	hist = model.fit({'main_input':train_set_x}, {'loss_01_layer':np.zeros((train_set_y.shape[0],1)), 'softmax_layer':train_set_y},
					 batch_size=256, shuffle=True, nb_epoch=NB_EPOCHS, validation_split=0.1, callbacks=[earlystop,mcp])

def model_predict(train_set_x, test_set_x, gallery_set_y, query_set_y):
	global WEIGHTS_SAVE_PATH, WEIGHTS_FILE_NAME
	if not WEIGHTS_FILE_NAME:
		print 'no weights_file, please add weights file!'
		return
	print 'predict start time: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
	model = build_DDN_net(HASH_NUM, SPLIT_NUM, REGULARIZER_PARAMS)
	model.load_weights(WEIGHTS_SAVE_PATH + WEIGHTS_FILE_NAME)
	Deepid_output = Model(input=model.get_layer('main_input').input,output=model.get_layer('A6').output)
	gallery_set_x = Deepid_output.predict(train_set_x)
	query_set_x = Deepid_output.predict(test_set_x)

	gallery_binary_x = T.sgn(gallery_set_x).eval()
	query_binary_x = T.sgn(query_set_x).eval()

	train_binary_x, train_data_y = gallery_binary_x, gallery_set_y
	train_data_y.shape = (gallery_set_y.shape[0], 1)
	test_binary_x, test_data_y = query_binary_x, query_set_y
	test_data_y.shape = (query_set_y.shape[0], 1)

	train_y_rep = repmat(train_data_y, 1, test_data_y.shape[0])
	test_y_rep = repmat(test_data_y.T, train_data_y.shape[0], 1)
	cateTrainTest = (train_y_rep == test_y_rep)
	train_data_y = train_data_y + 1
	test_data_y = test_data_y + 1

	train_data_y = np.asarray(train_data_y, dtype=int)
	test_data_y = np.asarray(test_data_y, dtype=int)

	B = compactbit(train_binary_x)
	tB = compactbit(test_binary_x)

	hammRadius = 2
	hammTrainTest = hammingDist(tB, B).T

	Ret = (hammTrainTest <= hammRadius + 0.000001)
	[Pre, Rec] = evaluate_macro(cateTrainTest, Ret)
	print 'Precision with Hamming radius_2 = ', Pre
	print 'Recall with Hamming radius_2 = ', Rec

	HammingRank = np.argsort(hammTrainTest, axis=0)
	[MAP, p_topN] = cat_apcal(train_data_y, test_data_y, HammingRank, TOP_K)
	print 'MAP with Hamming Ranking = ', MAP
	print 'Precision of top %d returned = %f ' % (TOP_K, p_topN)
	print 'predict finish time: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

if __name__ == '__main__':

	WEIGHTS_SAVE_PATH = check_path_valid(WEIGHTS_SAVE_PATH)
	dataset = load_data_split_pickle((TEST_SET_PATH, TRAIN_SET_PATH))
	train_set_x, train_set_y = dataset[0]
	test_set_x, test_set_y = dataset[1]
	train_set_x = train_set_x.reshape((train_set_x.shape[0], 3, IMAGE_WIDTH, IMAGE_HEIGHT))
	test_set_x = test_set_x.reshape((test_set_x.shape[0], 3, IMAGE_WIDTH, IMAGE_HEIGHT))

	gallery_set_y = copy.deepcopy(train_set_y)
	train_set_y = np_utils.to_categorical(train_set_y, NB_CLASSES)
	train_set_x = deprocess_image(train_set_x)
	query_set_y = copy.deepcopy(test_set_y)
	test_set_y = np_utils.to_categorical(test_set_y, NB_CLASSES)
	test_set_x = deprocess_image(test_set_x)

	if RUN_OR_CHECK:
		model_train(train_set_x,train_set_y)
	else:
		model_predict(train_set_x,test_set_x,gallery_set_y,query_set_y)