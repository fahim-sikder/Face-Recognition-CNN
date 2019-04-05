from os import walk
from PIL import Image
from numpy import array
from random import shuffle
from matplotlib import pyplot as plt
import numpy as np



IMAGE_PATH = '/home/jalkana/Documents/custom_dataset_face/OUTOUT'


def image_info(path):
    img = Image.open(path )
    arr = array(img)

    return arr

def load_input_data():



    print ''
    file_name_list = []

    dc = dict()

    cn = 0

    for (dirpath, dirnames, filenames) in walk(IMAGE_PATH):
        if len(filenames) == 0:
            continue

        

        dc[dirpath] = filenames
        #print dirnames

    


    print len(dc)

    y_lable_dic = dict()
    x_lable_dic = dict()
    lav = dict()
    di = 0

    for ls in dc:

        if (ls in y_lable_dic):
            print 'repeted name'
            exit()
        else:
            y_lable_dic[ls] = di
            lav[di] = ls
            di += 1

        for p in dc[ls]:
            str = ls + '/' + p
            file_name_list.append(str)
            x_lable_dic[str] = ls



    shuffle(file_name_list)

    ln = len(file_name_list)

    a = np.zeros((ln, 128, 128, 3))
    b = np.zeros((ln, 5))

    a1 = image_info(file_name_list[0])
    #print (a1[0][0]), ' a1[][][]'

    cnt = 0
    for img_pth in file_name_list:
        x = image_info(img_pth)
        #print 'width',len( x[0]), '   ', len(x[0][0])

        a[cnt] = x/255.0

        po = y_lable_dic[ x_lable_dic[img_pth]]
        #print po , ' po'
        b[cnt][po] = 1

        cnt += 1
        #exit()

    data = a[0]

    #print 'file index : ', x_lable_dic[file_name_list[0]]

    #plt.imshow(data, interpolation='nearest')
    #plt.show()

    #print b[0]
    #print b[2]

    #print len(a), '  len of a'
    #print len(a[0]), '  len of a[]'
    #print len(a[0][0]), '  len of a[][]'
    #print (a[0][0][0]), '  len of a[][][]'

    return a,b, lav



