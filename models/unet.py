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


class unet(object):
    def __init__(self, rows=512, cols=512):
        self.rows = rows
        self.cols = cols
    def model(self,input):
        conv1 = conv_bn_layer(input=input, num_filters=64, filter_size=3, stride=1, act='relu')
        print('conv1     shape',conv1.shape)
        conv1 = conv_bn_layer(input=conv1, num_filters=64, filter_size=3, stride=1, act='relu')
        print('conv1     shape',conv1.shape)
        pool1 = fluid.layers.pool2d(input=conv1, pool_size=3, pool_stride=2, pool_padding=1,pool_type='max')
        print('pool1     shape',pool1.shape)

        conv2 = conv_bn_layer(input=pool1, num_filters=128, filter_size=3, stride=1, act='relu')
        print('conv2     shape',conv2.shape)
        conv2 = conv_bn_layer(input=conv2, num_filters=128, filter_size=3, stride=1, act='relu')
        print('conv2     shape',conv2.shape)
        pool2 = fluid.layers.pool2d(input=conv2, pool_size=3, pool_stride=2, pool_padding=1,pool_type='max')
        print('pool2     shape',pool2.shape)

        conv3 = conv_bn_layer(input=pool2, num_filters=256, filter_size=3, stride=1, act='relu')
        print('conv3     shape',conv3.shape)
        conv3 = conv_bn_layer(input=conv3, num_filters=256, filter_size=3, stride=1, act='relu')
        print('conv3     shape',conv3.shape)
        pool3 = fluid.layers.pool2d(input=conv3, pool_size=3, pool_stride=2, pool_padding=1,pool_type='max')
        print('pool3     shape',pool3.shape)

        conv4 = conv_bn_layer(input=pool3, num_filters=512, filter_size=3, stride=1, act='relu')
        print('conv4     shape',conv4.shape)
        conv4 = conv_bn_layer(input=conv4, num_filters=512, filter_size=3, stride=1, act='relu')
        print('conv4     shape',conv4.shape)
        drop4 = fluid.layers.dropout(conv4, dropout_prob=0.5)
        print("drop4     shape",drop4.shape)
        pool4 = fluid.layers.pool2d(input=drop4, pool_size=3, pool_stride=2, pool_padding=1,pool_type='max')
        print('pool4     shape',pool4.shape)
        conv5 = conv_bn_layer(input=pool4, num_filters=1024, filter_size=3, stride=1, act='relu')
        print('conv5     shape',conv5.shape)
        conv5 = conv_bn_layer(input=conv5, num_filters=1024, filter_size=3, stride=1, act='relu')
        print('conv5     shape',conv5.shape)
        drop5 = fluid.layers.dropout(conv5, dropout_prob=0.5)
        print('drop5     shape',drop5.shape)

        deconv6 = deconv_bn_layer(drop5, num_filters = 512, filter_size=4, stride=2, act='relu')
        print("deconv6   shape",deconv6.shape)
        merge6 = fluid.layers.concat([drop4,deconv6], axis = 1) #merge in channel
        print('merge6    shape',merge6.shape)
        conv6 = conv_bn_layer(input=merge6, num_filters=512, filter_size=3, stride=1, act='relu')
        print('conv6     shape',conv6.shape)
        conv6 = conv_bn_layer(input=conv6, num_filters=512, filter_size=3, stride=1, act='relu')
        print('conv6     shape',conv6.shape)

        deconv7 = deconv_bn_layer(conv6, num_filters = 256, filter_size=4, stride=2,act='relu')
        print("deconv7   shape",deconv7.shape)
        merge7 = fluid.layers.concat([conv3,deconv7], axis = 1) #merge in channel
        print('merge7    shape',merge7.shape)
        conv7 = conv_bn_layer(input=merge7, num_filters=256, filter_size=3, stride=1, act='relu')
        print('conv7     shape',conv7.shape)
        conv7 = conv_bn_layer(input=conv7, num_filters=256, filter_size=3, stride=1, act='relu')
        print('conv7     shape',conv7.shape)

        deconv8 = deconv_bn_layer(conv7, num_filters = 128, filter_size=4, stride=2, act='relu')
        print("deconv8   shape",deconv8.shape)
        merge8 = fluid.layers.concat([conv2,deconv8], axis = 1) #merge in channel
        print('merge8    shape',merge8.shape)
        conv8 = conv_bn_layer(input=merge8, num_filters=128, filter_size=3, stride=1, act='relu')
        print('conv8     shape',conv8.shape)
        conv8 = conv_bn_layer(input=conv8, num_filters=128, filter_size=3, stride=1, act='relu')
        print('conv8     shape',conv8.shape)

        deconv9 = deconv_bn_layer(conv8, num_filters = 64, filter_size=4, stride=2,act='relu')
        print("deconv9   shape",deconv9.shape)
        merge9 = fluid.layers.concat([conv1,deconv9], axis = 1) #merge in channel
        print('merge9    shape',merge9.shape)
        conv9 = conv_bn_layer(input=merge9, num_filters=64, filter_size=3, stride=1, act='relu')
        print('conv9     shape',conv9.shape)
        conv9 = conv_bn_layer(input=conv9, num_filters=64, filter_size=3, stride=1, act='relu')
        print('conv9     shape',conv9.shape)
        conv9 = conv_bn_layer(input=conv9, num_filters=32, filter_size=3, stride=1, act='relu')
        print('conv9     shape',conv9.shape)
        conv10 = conv_bn_layer(input=conv9, num_filters=9, filter_size=1, stride=1, act='relu')

        conv10_transpose = fluid.layers.transpose(x=conv10, perm=[0, 2, 3, 1])
        modelOut = fluid.layers.reshape(conv10_transpose, shape=[-1, 9])
        modelOut = fluid.layers.softmax(modelOut)
        print('modelOut.shape == ',modelOut.shape)
      
        print("----------------------------------------------------------------")
        return modelOut