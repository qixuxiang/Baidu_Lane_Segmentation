#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
from utils import *
import utils
import contextlib
import os
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class multires_unet(object):
    def __init__(self, rows=512, cols=512):
        self.rows = rows
        self.cols = cols
    def ChannelSE(self, input, num_channels, reduction_ratio=16):
        """
        Squeeze and Excitation block, reimplementation inspired by
            https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py
        """
        pool = fluid.layers.pool2d(
            input=input, pool_size=0, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(input=pool,
                                size=num_channels // reduction_ratio,
                                act='relu',
                                param_attr=fluid.param_attr.ParamAttr(
                                    initializer=fluid.initializer.Uniform(
                                        -stdv, stdv)))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(input=squeeze,
                                    size=num_channels,
                                    act='sigmoid',
                                    param_attr=fluid.param_attr.ParamAttr(
                                        initializer=fluid.initializer.Uniform(
                                            -stdv, stdv)))
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale

    def SpatialSE(self, input):
        """
        Spatial squeeze and excitation block (applied across spatial dimensions)
        """
        conv = conv_bn_layer(input=input, num_filters=input.shape[1],
        filter_size=1, stride=1, act='sigmoid')
        conv = fluid.layers.elementwise_mul(x=input, y=conv, axis=0)
        return conv

    def scSE_block(self, x, channels):
        '''
        Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
        https://arxiv.org/abs/1803.02579
        '''
        cse = self.ChannelSE(input = x, num_channels = channels)
        sse = self.SpatialSE(input = x)
        x = fluid.layers.elementwise_add(x=cse, y=sse)
        return x

    def multi_res_block(self, inputs,filter_size1,filter_size2,filter_size3,filter_size4):

        conv1 = conv_bn_layer(
        input=inputs, num_filters=filter_size1, filter_size=3, stride=1, act='relu')

        conv2 = conv_bn_layer(
        input=conv1, num_filters=filter_size2, filter_size=3, stride=1, act='relu')

        conv3 = conv_bn_layer(
        input=conv2, num_filters=filter_size3, filter_size=3, stride=1, act='relu')

        conv = conv_bn_layer(
        input=inputs, num_filters=filter_size4, filter_size=1, stride=1, act='relu')

        concat = fluid.layers.concat([conv1, conv2, conv3], axis = 1) #merge in channel

        add = fluid.layers.elementwise_add(concat,y=conv)

        return add

    def res_path(self, inputs,filter_size,path_number):
        def block(x,fl):
            conv1 = conv_bn_layer(
            input=inputs, num_filters=filter_size, filter_size=3, stride=1, act='relu')

            conv2 = conv_bn_layer(
            input=inputs, num_filters=filter_size, filter_size=1, stride=1, act='relu')

            add = fluid.layers.elementwise_add(conv1,conv2)

            return add

        cnn = block(inputs, filter_size)
        if path_number <= 3:
            cnn = block(cnn,filter_size)
            if path_number <= 2:
                cnn = block(cnn,filter_size)
                if path_number <= 1:
                    cnn = block(cnn,filter_size)

        return cnn

    def model(self, inputs):
        res_block1 = self.multi_res_block(inputs,8,17,26,51)
        res_block1 = self.scSE_block(res_block1, res_block1.shape[1])


        pool1 = fluid.layers.pool2d(
            input=res_block1, pool_size=3, pool_stride=2, pool_padding=1, \
            pool_type='max')
        print(pool1.shape)


        res_block2 = self.multi_res_block(pool1,17,35,53,105)
        res_block2 = self.scSE_block(res_block2, res_block2.shape[1])


        pool2 = fluid.layers.pool2d(
            input=res_block2, pool_size=3, pool_stride=2, pool_padding=1, \
            pool_type='max')
        print(pool2.shape)

        res_block3 = self.multi_res_block(pool2,31,72,106,209)
        res_block3 = self.scSE_block(res_block3, res_block3.shape[1])
        pool3 = fluid.layers.pool2d(
            input=res_block3, pool_size=3, pool_stride=2, pool_padding=1, \
            pool_type='max')
        print(pool3.shape)

        res_block4 = self.multi_res_block(pool3,71,142,213,426)
        res_block4 = self.scSE_block(res_block4, res_block4.shape[1])

        pool4 = fluid.layers.pool2d(
            input=res_block4, pool_size=3, pool_stride=2, pool_padding=1, \
            pool_type='max')
        print(pool4.shape)

        res_block5 = self.multi_res_block(pool4,142,284,427,853)
        
        upsample = deconv_bn_layer(res_block5, num_filters = 853, filter_size=4, stride=2,
                        act='relu')
        print("upsample.shape",upsample.shape)

        res_path4 = self.res_path(res_block4,256,4)

        concat = fluid.layers.concat([upsample,res_path4], axis = 1) #merge in channel
        print("concat.shape",concat.shape)
        res_block6 = self.multi_res_block(concat,71,142,213,426)
        res_block6 = self.scSE_block(res_block6, res_block6.shape[1])
        upsample = deconv_bn_layer(res_block6, num_filters = 426, filter_size=4, stride=2,
                        act='relu')
        print("upsample.shape",upsample.shape)

        res_path3 = self.res_path(res_block3,128,3)
        concat = fluid.layers.concat([upsample,res_path3], axis = 1) #merge in channel
        print("concat.shape",concat.shape)

        res_block7 = self.multi_res_block(concat,31,72,106,209)
        res_block7 = self.scSE_block(res_block7, res_block7.shape[1])
        upsample = deconv_bn_layer(res_block7, num_filters = 209, filter_size=4, stride=2,
                        act='relu')
        print("upsample.shape",upsample.shape)

        res_path2 = self.res_path(res_block2,64,2)
        concat = fluid.layers.concat([upsample,res_path2], axis = 1) #merge in channel
        print("concat.shape",concat.shape)

        res_block8 = self.multi_res_block(concat,17,35,53,105)
        res_block8 = self.scSE_block(res_block8, res_block8.shape[1])
        upsample = deconv_bn_layer(res_block8, num_filters = 105, filter_size=4, stride=2,
                        act='relu')
        print("upsample.shape",upsample.shape)

        res_path1 = self.res_path(res_block1,32,1)
        concat = fluid.layers.concat([upsample,res_path1], axis = 1) #merge in channel
        print("concat.shape",concat.shape)

        res_block9 = self.multi_res_block(concat,8,17,26,51)
        res_block9 = self.scSE_block(res_block9, res_block9.shape[1])
        conv9 =  conv_bn_layer(
            input=res_block9, num_filters=9, filter_size=1, stride=1, act='relu')

        conv9_transpose = fluid.layers.transpose(x=conv9, perm=[0, 2, 3, 1])
        modelOut = fluid.layers.reshape(conv9_transpose, shape=[-1, 9])
        modelOut = fluid.layers.softmax(modelOut)
        return modelOut