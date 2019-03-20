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

from reader import TrainDataReader
import reader

os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.99'


pretrain_model = 0 #1 means pretrain_model
total_step = 150000
path = os.getcwd()

def saveImage(img,path):

    label_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    label_rgb[img == 0] =  [  0,   0,   0]
    label_rgb[img == 1] =  [ 70, 130, 180]
    label_rgb[img == 2] =  [119,  11,  32]
    label_rgb[img == 3] =  [220, 220,   0]
    label_rgb[img == 4] =  [102, 102, 156]
    label_rgb[img == 5] =  [128,  64, 128]
    label_rgb[img == 6] =  [190, 153, 153]
    label_rgb[img == 7] =  [128, 128,   0]
    label_rgb[img == 8] =  [255, 128,   0]
    img = cv2.resize(label_rgb, (4000,4000), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, label_rgb)

def categorical_crossentropy(label, pred):
    pred = fluid.layers.clip(x=pred, min=1e-7, max=1. - 1e-7)
    label = fluid.layers.reshape(label, shape=[-1,9])
    print('label.shape-----||||------pred.shape',label.shape,pred.shape)
    loss = fluid.layers.cross_entropy(input=pred, label=label,soft_label=True)
    #loss = fluid.layers.softmax_with_cross_entropy(logits=pred,label=label,ignore_index=0)
    loss = fluid.layers.reduce_mean(loss)
    loss = fluid.layers.clip(x=loss, min=3e-7, max=3. - 1e-7)
    return loss

def focal_loss(target,output):
    output = fluid.layers.clip(x=output, min=1e-7, max=1. - 1e-7)
    target = fluid.layers.reshape(target, shape=[-1,9])
    log = fluid.layers.log(output)
    log = fluid.layers.abs(log)
    log = fluid.layers.elementwise_mul(target,log)
    one = fluid.layers.fill_constant_batch_size_like(input=output, shape=output.shape, value=1.0, dtype='float32')
    one_sub_pt = fluid.layers.elementwise_sub(one,output)
    one_sub_pt_pow = fluid.layers.elementwise_mul(one_sub_pt,one_sub_pt)
    totalCost = fluid.layers.elementwise_mul(one_sub_pt_pow,log)
    loss = fluid.layers.reduce_mean(totalCost)
    loss = fluid.layers.clip(x=loss, min=1e-7, max=1. - 1e-7)
    return loss

def create_iou(predict, label, num_classes):
    print('iou_label.shape',predict.shape,label.shape)
    label = fluid.layers.reshape(label, shape=[-1,9])
    print('iou_label.shape',predict.shape,label.shape)
    label = fluid.layers.argmax(label, axis=1).astype('int64')
    predict = fluid.layers.argmax(predict, axis=1).astype('int64')
    print('iou_label.shape',predict.shape,label.shape)
    iou, out_wrong, out_right = fluid.layers.mean_iou(predict, label, num_classes)

    return iou,out_wrong,out_right

def cal_mean_iou(wrong, correct):
    sum = wrong + correct
    true_num = (sum != 0).sum()
    for i in range(len(sum)):
        if sum[i] == 0:
            sum[i] = 1
    return (correct.astype("float64") / sum).sum() / true_num

def create_reader(rows=1024,cols=1024):

    LaneDataset = reader.TrainDataReader
    #dataset = LaneDataset("/media/airobot/docs/BaiduDatas/ApolloDatas/", 'train',
    #                        rows=1024, cols=1024)
    dataset = LaneDataset(path + "/data/ApolloDatas/", 'train',
                            rows=1024, cols=1024)
    return dataset

def create_model(model='',image_shape=[1024,1024],class_num=9):

    train_image = fluid.layers.data(name='img', shape=[3] + image_shape, dtype='float32')
    train_label = fluid.layers.data(name='label', shape=image_shape + [9],dtype='float32')

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

    loss = categorical_crossentropy(train_label,predict)
    iou,out_wrong,out_right = create_iou(predict,train_label,class_num)

    return predict,loss,iou

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
        fluid.io.load_params(executor=exe, dirname="", filename=path+'/params/deeplabv3p_ours.params', main_program=program)


def save_model(exe,program,model=''):
    if model == 'unet':
        fluid.io.save_params(executor=exe, dirname="", filename=path+'/params/unet.params', main_program=program)
    if model == 'deeplab_v3p':
        fluid.io.save_params(executor=exe, dirname="", filename=path+'/params/deeplab_v3p.params', main_program=program)
    if model == 'pannet':
        fluid.io.save_params(executor=exe, dirname="", filename=path+'/params/pannet.params', main_program=program)
    if model == 'dense_unet':
        fluid.io.save_params(executor=exe, dirname="", filename=path+'/params/dense_unet.params', main_program=program)
    if model == 'multires_unet':
        fluid.io.loadsave_params_params(executor=exe, dirname="", filename=path+'/params/multires_unet.params', main_program=program)
    if model == 'deeplabv3p_ours':
        fluid.io.loadsave_params_params(executor=exe, dirname="", filename=path+'/params/deeplabv3p_ours.params', main_program=program)




def train(model):

    predict,loss,iou = create_model(model=model)
    optimizer = fluid.optimizer.Adam(learning_rate=1e-4)
    optimizer.minimize(loss)
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())
    fluid.memory_optimize(fluid.default_main_program(),print_log=False, skip_opt_set=set([loss.name,predict.name]))

    if pretrain_model:
        load_model(exe,fluid.default_main_program(),model=model)
        print("load model succeed")
    else:
        print("load succeed")

    def trainLoop():
        batches = DataSet.get_batch_generator(1, total_step)
        iou_count = 0
        mean_iou  = 0
        iou_sum   = 0
        for i, imgs, labels, names in batches:
            preTime = time.time()
            result = exe.run(fluid.default_main_program(),
                            feed={'img': imgs,
                                    'label': labels},
                           fetch_list=[loss,predict,iou])
            nowTime = time.time()

            iou_sum   += result[2]
            iou_count += 1
            mean_iou = iou_sum/iou_count

            print('                                                         iou = ',result[2],'mean_iou = ',mean_iou)
            
            if iou_count % 1000 == 0:
                iou_count = 0
                iou_sum   = 0

            if i % 1000 == 0 and i!= 0:
                print("Model saved")
                save_model(exe,fluid.default_main_program(),model=model)
            
            if i % 10 ==0:
                train_path = path+'/train.png'
                picture = result[1]
                picture = np.argmax(picture,axis=-1)
                picture = picture.reshape((1024,1024))
                saveImage(picture,train_path)
                label_path = path+'/trainlabel.png'
                train_lab = np.argmax(labels[0],axis=2)
                saveImage(train_lab,label_path)
                
            if i % 20 == 0:
                argmax = np.argmax(result[1],axis=1)
                abc = Counter(argmax)
                print('                                        ',abc)

            if i % 2 == 0:
                print("step {:d},loss {:.6f},step_time: {:.3f}".format(
                    i,result[0][0],nowTime - preTime))

    trainLoop()


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='')
    parse.add_argument('--model', help='model name', nargs='?')
    args = parse.parse_args()
    model = args.model
    DataSet = create_reader(model)
    train(model)