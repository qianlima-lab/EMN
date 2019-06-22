from __future__ import print_function
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from matplotlib.pyplot import savefig
import random
random.seed(2345)
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import concatenate
from keras import regularizers
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
import ucr_reading
from ucr_reading import *

dir_path = 'UCR_TS_Archive_2015'
list_dir = os.listdir(dir_path)
index = list_dir.index('ECG200')
number_res=32
IS=0.1
SR=0.9
SP=0.7
nb_epoch =500
batch_size=25
nb_filter=120
ratio=[0.6,0.7]

print('Loading data...')	
train_echoes, train_y, test_echoes, test_y, dataset_name, n_res, len_series, IS, SR, SP = run_loading(index=index, n_res =number_res,IS=IS,SR=SR,SP=SP)
print('Transfering label...')
train_y, nums_class = transfer_labels(train_y)
test_y, _ = transfer_labels(test_y)
train_y = np_utils.to_categorical(train_y, nums_class)
test_y = np_utils.to_categorical(test_y, nums_class)
nb_class = nums_class
nb_sample_train = np.shape(train_echoes)[0]
nb_sample_test = np.shape(test_echoes)[0]
test_data = np.reshape(test_echoes,(nb_sample_test, 1, len_series, n_res))
test_labels = test_y
L_train = [x_train for x_train in range(nb_sample_train)]
np.random.shuffle(L_train)
train_data = np.zeros((nb_sample_train, 1, len_series, n_res))
train_label = np.zeros((nb_sample_train, nb_class))
for m in range(nb_sample_train):
	train_data[m,0,:,:] = train_echoes[L_train[m],:,:]
	train_label[m,:] = train_y[L_train[m],:]

input_shape = (1, len_series, n_res)
nb_row=[np.int(ratio[0]*len_series),np.int(ratio[1]*len_series)]
nb_col = input_shape[2]
kernel_initializer = 'lecun_uniform'
activation = 'relu'
padding = 'valid'
strides = (1, 1)
data_format='channels_first'
optimizer = 'adam'
loss = ['binary_crossentropy', 'categorical_crossentropy']
verbose = 1
#model
input = Input(shape = input_shape)
convs = []
for j in range(len(nb_row)):
	conv = Conv2D(nb_filter, (nb_row[j], nb_col), kernel_initializer = kernel_initializer, activation = activation, 
	padding = padding, strides = strides, data_format = data_format)(input)
	conv = GlobalMaxPooling2D(data_format = data_format)(conv)
	convs.append(conv)
body_feature = concatenate(convs,name='concat_layer')
#body_feature = Dense(64, kernel_initializer = kernel_initializer, activation = activation)(body_feature)
#body_feature = Dense(128, kernel_initializer = kernel_initializer, activation = activation)(body_feature)
body_feature = Dropout(0.25)(body_feature)
output = Dense(nb_class, kernel_initializer = kernel_initializer, activation = 'softmax',name = 'dense_output')(body_feature)
model = Model(input = input, output = output)
model.summary()
model.compile(optimizer = optimizer, loss = loss[1], metrics = ['accuracy'])
history = model.fit(train_data, train_label, batch_size = batch_size, 
	epochs = nb_epoch, verbose = verbose, 
	validation_data = (test_data, test_labels))
log = pd.DataFrame(history.history)
minloss_acc=log.loc[log['loss'].idxmin]['val_acc']

plt.figure(figsize=(9,3))
plt.plot(history.history['acc'],linewidth=0.5)
plt.plot(history.history['val_acc'],linewidth=0.5)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
savefig('./plot/'+dataset_name+'_acc.pdf')
plt.show()
print('batch:',batch_size)
print('filter:,',nb_filter)
print('nb_row:',ratio)
print('IS :', IS)
print('SR:', SR)
print('SP:', SP)
print('Size:', number_res)
print('dataset:',dataset_name)
print('accuracy:',minloss_acc)

