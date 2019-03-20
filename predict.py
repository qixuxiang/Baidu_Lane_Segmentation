#-*- coding:utf-8 -*-
import os
import argparse
import numpy as np
import time
import sys
import argparse
import paddle
import paddle.fluid as fluid
import cv2
from collections import Counter

from models.unet import unet
from models.PAN import pannet
from models.dense_unet import dense_unet
from models.multires_unet import multires_unet
from models.deeplabv3p import deeplab_v3p
from models.deeplabv3p_ours import deeplabv3p_ours

from reader import TestDataReader
import reader

path = os.getcwd()

def create_reader(rows=1024,cols=1024):

    LaneDataset = reader.TestDataReader
    #dataset = LaneDataset("/media/airobot/docs/BaiduDatas/apolloscape/apolloscape/", 'test',
    #                        rows=1024, cols=1024)
    dataset = LaneDataset(path + "/data/ApolloDatas/", 'test',
                        rows=1024, cols=1024)
    return dataset

def create_model(model='',image_shape=[1024,1024],class_num=9):

    train_image = fluid.layers.data(name='img', shape=[3] + image_shape, dtype='float32')

    if model == 'unet':
        predict = unet().model(train_image)
    if model == 'deeplab_v3p':
        predict = deeplab_v3p().model(train_image)
    if model == 'pannet':
        predict = pannet().model(train_image)
    if model == 'dense_unet':
        predict = dense_unet().model(train_image)
    if model == 'multires_unet':
        predict = multires_unet().model(train_image)
    if model == 'deeplabv3p_ours':
        predict = deeplabv3p_ours().model(train_image)

    return predict

def get_M_Minv():
    # 左上、右上、左下、右下
    src = np.float32([[800, 730], [2583, 730], [0, 1709], [3383, 1709]])
    dst = np.float32([[0, 0], [3999,0], [1300, 3999], [2700, 3999]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    return M,Minv

M,Minv = get_M_Minv()

def saveImage(img,path):
    #img = np.zeros((imageShape[0],imageShape[1], 1), dtype=np.uint8)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] == 0:
               img[i][j] = 0
            elif img[i][j] == 1:
                img[i][j] = 200
            elif img[i][j] == 2:
                img[i][j] = 203
            elif img[i][j] == 3:
                img[i][j] = 217
            elif img[i][j] == 4:
                img[i][j] = 218
            elif img[i][j] == 5:
                img[i][j] = 210
            elif img[i][j] == 6:
                img[i][j] = 214
            elif img[i][j] == 7:
                img[i][j] = 220
            elif img[i][j] == 8:
                img[i][j] = 205

    img = cv2.resize(img, (4000,4000), interpolation=cv2.INTER_NEAREST)
    img = cv2.warpPerspective(img, Minv, (3384, 1710),flags=cv2.INTER_NEAREST) 
    #img = cv2.resize(img, (3384, 1710), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, img)

def load_model(exe,program,model=''):
        if model == 'unet':
            fluid.io.load_params(executor=exe, dirname="", filename=path+'/params/unet.params', main_program=program)
        if model == 'deeplab_v3p':
            fluid.io.load_params(executor=exe, dirname="", filename=path+'/params/deeplab_v3p.params', main_program=program)
        if model == 'pannet':
            fluid.io.load_params(executor=exe, dirname="", filename=path+'/params/pannet.params', main_program=program)
        if model == 'dense_unet':
            fluid.io.load_params(executor=exe, dirname="", filename=path+'/params/dense_unet.params', main_program=program)
        if model == 'multires_unet':
            fluid.io.load_params(executor=exe, dirname="", filename=path+'/params/multires_unet.params', main_program=program)
        if model == 'deeplabv3p_ours':
            fluid.io.loadsave_params_params(executor=exe, dirname="", filename=path+'/params/deeplabv3p_ours.params', main_program=program)




if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='')
    parse.add_argument('--model', help='model name', nargs='?')
    args = parse.parse_args()
    model = args.model

    DataSet = create_reader(model)

    predict = create_model(model=model)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    fluid.memory_optimize(fluid.default_main_program())
    load_model(exe,fluid.default_main_program(),model=model)

    batches = DataSet.get_batch_generator(1, 1234)
    for i, imgs, names in batches:

        result = exe.run(fluid.default_main_program(),
                        feed={'img': imgs},
                        fetch_list=[predict])
        print(i)
        path = path+'data/unet/test/ColorImage/' + names[0].split("image/")[1]
        picture = np.argmax(result[0],axis=1)
        picture = picture.reshape((1024,1024))
        saveImage(picture,path)