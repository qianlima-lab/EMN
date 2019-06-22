import os
import scipy as sp
import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import *
import matplotlib.pyplot as plt
from matplotlib import colors

def list_ucr():
	dir_path = './UCR_TS_Archive_2015'
	list_dir = os.listdir(dir_path)
	print list_dir

def loading_ucr(index=1):
	dir_path = 'UCR_TS_Archive_2015'
	list_dir = os.listdir(dir_path)
	dataset = list_dir[index]
	train_data = dir_path+'/'+dataset+'/'+dataset+'_TRAIN'
	test_data = dir_path+'/'+dataset+'/'+dataset+'_TEST'
	train_x, train_y = open_file(train_data,dataset,'train')
	test_x, test_y   = open_file(test_data,dataset,'test')
	return train_x, train_y, test_x, test_y, dataset

def open_file(path_data,dataset, info):
	data_x = []
	data_y = []
	count = 0;
	for line in open(path_data):
		count = count + 1
		#print '%s reading %03d-th row in dataset - %s'%(info, count, dataset)
		row = [[np.float32(x)] for x in line.strip().split(',')]
		label = np.int32(row[0])
		row = np.array(row[1:])
		row_shape = np.shape(row)
		# row = row.reshape((row_shape[1],row_shape[0]))
		row_mean = np.mean(row[0:])
		#data_x.append(row)
		data_x.append(row-np.kron(np.ones((row_shape[0],row_shape[1])),row_mean))
		data_y.append(label[0])
		if count == 1:
			len_series = np.shape(data_x)[1]		
	return  data_x, data_y

""" Echo State Network """
class reservoir_layer(object):
	def __init__(self, rng, n_in, n_res, IS, SR, sparsity, leakyrate, use_bias=False):
		self.n_in = n_in
		self.n_res = n_res
		self.IS = IS
		self.SR = SR
		self.sparsity = sparsity
		self.leakyrate = leakyrate
		self.use_bias = use_bias
		self.W_in = 2*np.array(np.random.random(size=(n_res, n_in)))-1
		W_res_temp = sp.sparse.rand(self.n_res, self.n_res, self.sparsity)
		vals, vecs = sp.sparse.linalg.eigsh(W_res_temp, k=1)
		self.W_res = self.SR * W_res_temp / vals[0]
		b_bound = 0.1
		self.b = 2*b_bound*np.random.random(size=(self.n_res, 1))-b_bound
	def update_states(self, data, dataset, string):
		n_forget_steps = 0
		nums_sample = np.shape(data)[0]
		nums_frame = np.shape(data)[1]
		echo_states = np.empty((nums_sample,(nums_frame-n_forget_steps), self.n_res), np.float32)
		for i_sample in range(nums_sample):
			series = data[i_sample]
			print 'create echo-states of %4d-th %s sample in %s'%(i_sample, string, dataset)
			collect_states = np.zeros((nums_frame-n_forget_steps, self.n_res))
			x = np.zeros((self.n_res, 1))
			for t in range(nums_frame):
				#print "sample %03d for %03d th time stamp processed ......" % (i_sample+1, t+1)
				u_t = np.asarray([series[t,:]]).T
				if self.use_bias:
					xUpd =  np.tanh(np.dot(self.W_in, self.IS*u_t) + np.dot(self.W_res.toarray(), x) + self.b)
				else:        
					xUpd =  np.tanh(np.dot(self.W_in, self.IS*u_t) + np.dot(self.W_res.toarray(), x))
				x = (1-self.leakyrate)*x + self.leakyrate*xUpd
				if t >= n_forget_steps:
					collect_states[t-n_forget_steps,:] = x.T
			collect_states = np.asarray(collect_states)
			echo_states[i_sample] = collect_states
		return echo_states

def run_loading(index, n_res, IS, SR, SP):
	train_x, train_y, test_x, test_y, dataset_name = loading_ucr(index=index)
	nums_train, len_series =  np.shape(train_x)[0], np.shape(train_x)[1]
	nums_test = np.shape(test_x)[0]
	rng = np.random.RandomState(1882517)
	n_res = n_res
	IS = IS
	SR = SR
	SP = SP
	LK = 1
	escnn = reservoir_layer(rng, n_in=1, n_res=n_res, IS=IS, SR=SR, sparsity=SP, leakyrate=LK, use_bias=False)
	train_echoes = escnn.update_states(train_x, dataset_name, 'train')
	test_echoes = escnn.update_states(test_x, dataset_name, 'test')
	return train_echoes, train_y, test_echoes, test_y, dataset_name, n_res, len_series, IS, SR, SP 

def transfer_labels(labels):
	indexes = np.unique(labels)
	num_classes = indexes.shape[0]
	num_samples = np.shape(labels)[0]
	for i in range(num_samples):
		new_label = np.argwhere(indexes == labels[i])[0][0]
		labels[i] = new_label
	return labels, num_classes

# def transfer_labels(train_labels, test_labels):
	# nums_train = len(train_labels)
	# nums_test = len(test_labels)
	# indexes = np.unique(train_labels)
	# print indexes
	# min_class =  np.min(indexes)
	# nums_class = len(indexes) 
	# if (max(indexes) != nums_class) and (min_class != -1) and (min_class != 0):
		# train_labels, test_labels = label_modified(train_labels, test_labels, indexes, nums_class) 
	# if min_class == 1:
		# for i in range(nums_train):
			# if train_labels[i]==nums_class:
				# train_labels[i] = 0  
		# for i in range(nums_test):
			# if test_labels[i]==nums_class:
				# test_labels[i] = 0
	# elif min_class == -1:
		# for i in range(nums_train):
			# if train_labels[i]==-1:
				# train_labels[i] = 0
		# for i in range(nums_test):
			# if test_labels[i]==-1:
				# test_labels[i] = 0
	# print train_labels
	# print np.unique(train_labels)
	# return train_labels, test_labels, nums_class

# def label_modified(train_labels, test_labels, indexes, nums_class):
	# nums_train = len(train_labels)
	# nums_test = len(test_labels)
	# max_class = max(indexes)
	# classes = np.zeros((max_class,1))
	# train_labels_new = []
	# test_labels_new = []
	# count = 0
	# for i in indexes:
		# classes[i-1] = count
		# count += 1
	# for i in range(nums_train):
		# train_labels_new.append(np.int(classes[train_labels[i]-1][0]))
	# for i in range(nums_test):
		# test_labels_new.append(np.int(classes[test_labels[i]-1][0]))
	# return train_labels_new, test_labels_new

if __name__ == '__main__':
	list_ucr()







