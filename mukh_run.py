# -*- coding: utf-8 -*-


from __future__ import division, print_function, absolute_import

from os import walk
from PIL import Image
from numpy import array
import tensorflow
import tflearn
import pickle
import  os
import pandas as pd
import matplotlib.pyplot as plt
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from matplotlib import pyplot as plt
from img_input_data import *


X, Y, lavle = load_input_data()




network = input_data(shape=[None, 32, 32, 3])

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





model.load('output/2016-12-26___11:00:59.tflearn')


def draw_ploat():
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.axis([0, 16, 0, 1.00])
    plt.grid(True)
    plt.show()

with open('output/2016-12-26___11:00:59.pickle', 'rb') as handle:
    b = pickle.load(handle)

    image_num = 3
    a = np.zeros((image_num, 128, 128, 3))




    x1 = image_info('/home/jalkana/Documents/custom_dataset_face/unseen_face/img3.jpg')
    x2 =  image_info('/home/jalkana/Documents/custom_dataset_face/unseen_face/img2.jpg')
    x3 =  image_info('/home/jalkana/Documents/custom_dataset_face/unseen_face/img1.jpg')





    person_tag = []
    for i in range(5):
        person_tag.append( os.path.basename(os.path.normpath(b[i])) )




    print (b)

    #for i in range(image_num):
       # a[i] = X[i]

    a[0] = x1 / 256.0
    a[1] = x2 / 256.0
    a[2] = x3 / 256.0

    res = (model.predict(a))

    i =0
    for r in res:
        i += 1
        print (r)
        x2 = (np.max(r))
        cnt = 0
        #r.append(1.0)
        #person_tag.append(' ')
        #draw_plot(r, person_tag )
        for t in r:
            if x2 == t:
                #print (b[cnt])


                if x2 > 0.7:
                    person_name = os.path.basename(os.path.normpath(b[cnt]))
                else:
                    person_name = 'SORRY.... YOU ARE UNKNOWN'

                print (cnt, ' th person. Confidence :', (x2 *100) , ' percent   person name: ',  person_name )
                confi =  person_name + '\nconfidence '  + str(x2 *100) + ' % '



                plt.imshow(a[i-1], interpolation='nearest')
               
                plt.title( confi)
                plt.show()
                break
            else:
                cnt += 1


        #print (r)
        print ('')
