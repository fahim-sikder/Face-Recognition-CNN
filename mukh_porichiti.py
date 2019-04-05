# -*- coding: utf-8 -*-


from __future__ import division, print_function, absolute_import

from os import walk
from PIL import Image
from numpy import array
import tensorflow
import tflearn
import pickle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from time import gmtime, strftime
from matplotlib import pyplot as plt
from img_input_data import *


X, Y, lavle = load_input_data()

print ('person name: ', len(X), ' level:', len(Y[0]) )



network = input_data(shape=[None, 128, 128, 3])

network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)

network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)

network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)

network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)

network = fully_connected(network, 2048, activation='tanh')
network = dropout(network, 0.5)

network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)


model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=3)

#model.load('output/2016-12-24___23:35:58.tflearn')

model.fit(X, Y, n_epoch=500, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='Face_porichiti_1')



OUT_PATH = 'output/'
tim_str = strftime("%Y-%m-%d___%H:%M:%S", gmtime())
OUT_FILE_NAME = OUT_PATH + tim_str

model.save(OUT_FILE_NAME+".tflearn")

with open(OUT_FILE_NAME+ '.pickle', 'wb') as handle:
    pickle.dump(lavle, handle, protocol=pickle.HIGHEST_PROTOCOL)


