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

class dense_unet(object):
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

    def DenseBlock(self, inputs, outdim):
        input_shape = inputs.shape
        bn = fluid.layers.batch_norm(input=inputs, epsilon=2e-05,fuse_with_relu=True)
        conv1 = conv_bn_layer(
        input=bn, num_filters=outdim, filter_size=3, stride=1, act='relu')
        if input_shape[1] != outdim:
            shortcut = conv_bn_layer(
            input=inputs, num_filters=outdim, filter_size=1, stride=1, act='relu')
        else:
            shortcut = inputs
        result1 = fluid.layers.elementwise_add(conv1, shortcut)

        bn = fluid.layers.batch_norm(input=result1, epsilon=2e-05,fuse_with_relu=True)
        conv2 = conv_bn_layer(
        input=bn, num_filters=outdim, filter_size=3, stride=1, act='relu')

        result = fluid.layers.elementwise_add(result1, conv2)

        result = fluid.layers.elementwise_add(result, shortcut)

        result = fluid.layers.relu(result)

        return result

    def model(self, inputs):
        conv1 = conv_bn_layer(
        input=inputs, num_filters=32, filter_size=1, stride=1, act='relu')
        conv1 = self.DenseBlock(conv1, 32)  # 48
        print("conv1.shape",conv1.shape)
        conv1 = self.scSE_block(conv1, conv1.shape[1])
        print("conv1.shape",conv1.shape)
        pool1 = fluid.layers.pool2d(
            input=conv1, pool_size=3, pool_stride=2, pool_padding=1, \
            pool_type='max')
        print("pool1.shape",pool1.shape)
        conv2 = self.DenseBlock(pool1, 64)  # 48
        conv2 = self.scSE_block(conv2, conv2.shape[1])
        print("conv2.shape",conv2.shape)
        pool2 = fluid.layers.pool2d(
            input=conv2, pool_size=3, pool_stride=2, pool_padding=1, \
            pool_type='max')
        print("pool2.shape",pool2.shape)
        conv3 = self.DenseBlock(pool2, 64)  # 48
        conv3 = self.scSE_block(conv2, conv3.shape[1])
        print("conv3.shape",conv3.shape)
        pool3 = fluid.layers.pool2d(
            input=conv3, pool_size=3, pool_stride=2, pool_padding=1, \
            pool_type='max')
        print("pool3.shape",pool3.shape)
        conv4 = self.DenseBlock(pool3, 64)  # 12
        print("conv4.shape",conv4.shape)

        up1 = deconv_bn_layer(conv4, num_filters = 64, filter_size=4, stride=2,
                        act='relu')
        print("up1.shape",up1.shape)
        up1 = fluid.layers.concat([up1, conv3], axis = 1) #merge in channel

        conv5 = self.DenseBlock(up1, 64)
        conv5 = self.scSE_block(conv5, conv5.shape[1])
        print("conv5.shape",conv5.shape)
        up2 = deconv_bn_layer(conv5, num_filters = 64, filter_size=3, stride=1,
                        act='relu')
        print("up2.shape",up2.shape)
        up2 = fluid.layers.concat([up2, conv2], axis = 1) #merge in channel
        conv6 = self.DenseBlock(up2, 64)
        conv6 = self.scSE_block(conv6, conv6.shape[1])
        print("conv6.shape",conv6.shape)

        up3 = deconv_bn_layer(conv6, num_filters = 32, filter_size=4, stride=2,
                        act='relu')
        print("up3.shape",up3.shape)
        up1 = fluid.layers.concat([up3, conv1], axis = 1) #merge in channel
        conv7 = self.DenseBlock(up3, 32)
        conv7 = self.scSE_block(conv7, conv7.shape[1])
        print("conv7.shape",conv7.shape)

        

        conv8 =  conv_bn_layer(
            input=conv7, num_filters=9, filter_size=1, stride=1, act='sigmoid')
        conv8_transpose = fluid.layers.transpose(x=conv8, perm=[0, 2, 3, 1])
        modelOut = fluid.layers.reshape(conv8_transpose, shape=[-1, 9])
        modelOut = fluid.layers.softmax(modelOut)
        return modelOut