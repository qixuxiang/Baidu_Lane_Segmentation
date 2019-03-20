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

class pannet(object):
    def __init__(self, rows=512, cols=512):
        self.rows = rows
        self.cols = cols
    def Inception_dilation(self, inputs, channels):
        conv3 = conv_bn_layer(input=inputs, num_filters=channels,
        filter_size=3, stride=1, dilation=1, act='relu')
        print("conv3.shape----------",conv3.shape)

        conv5 = conv_bn_layer(input=inputs, num_filters=channels,
        filter_size=3, stride=1, dilation=2, act='relu')
        print("conv5.shape----------",conv5.shape)
        conv7 = conv_bn_layer(input=inputs, num_filters=channels,
        filter_size=3, stride=1, dilation=4, act='relu')
        print("conv7.shape----------",conv7.shape)
        conv9 = conv_bn_layer(input=inputs, num_filters=channels,
        filter_size=3, stride=1, dilation=6, act='relu')
        print("conv9.shape----------",conv9.shape)

        merge2 = fluid.layers.concat([conv3, conv5, conv7, conv9], axis = 1)
        return merge2

    def FeaturePyramidAttention(self, inputs, channels):
        conv1 = conv_bn_layer(input=inputs, num_filters=channels,
        filter_size=1, stride=1,dilation=1, act='relu')

        conv7 = conv_bn_layer(input=inputs, num_filters=channels,
        filter_size=3, stride=1, dilation=4, act='relu')
        print("before pooling,",conv7.shape)

        pool1 = fluid.layers.pool2d(conv7, pool_size=4, pool_type='max',
                pool_stride=4)
        print("after pooling,",pool1.shape)
        conv5 = conv_bn_layer(input=pool1, num_filters=channels,
        filter_size=3, stride=1, dilation=3, act='relu')
        pool2 = fluid.layers.pool2d(conv5, pool_size=4, pool_type='max',
                pool_stride=4)

        conv3 = conv_bn_layer(input=pool2, num_filters=channels,
        filter_size=3, stride=1, dilation=2, act='relu')
        pool3 = fluid.layers.pool2d(conv3, pool_size=4, pool_type='max',
                pool_stride=4)
        conv2 = conv_bn_layer(input=pool3, num_filters=channels,
        filter_size=3, stride=1, dilation=1, act='relu')

        up1 = fluid.layers.resize_bilinear(input = conv2,scale = 4)
        up1 = conv_bn_layer(input=up1, num_filters=channels,filter_size=1,
        stride=1,dilation=1, act='relu')
        up1 = fluid.layers.concat([up1, conv3], axis = 1)

        up2 = fluid.layers.resize_bilinear(input = up1,scale = 4)
        up2 = conv_bn_layer(input=up2, num_filters=channels,filter_size=1,
        stride=1,dilation=1, act='relu')
        up2 = fluid.layers.concat([up2, conv5], axis = 1)

        up3 = fluid.layers.resize_bilinear(input = up2,scale = 4)
        up3 = conv_bn_layer(input=up3, num_filters=channels,filter_size=1,
        stride=1,dilation=1, act='relu')
        up3 = fluid.layers.concat([up3, conv7], axis = 1)
        out = fluid.layers.concat([up3, conv1], axis = 1)
        return out

    def GlobalAttentionUpsample(self, inputs_low, inputs_high, channels):
        #inputs_low：低层次信息输入
        #inputs_high：高层次信息输入
        print('inputs_high.shape---------',inputs_high.shape)
        conv3 = conv_bn_layer(input=inputs_low, num_filters=3*channels,
        filter_size=3, stride=1,dilation=1, act='relu')
        gap = fluid.layers.pool2d(inputs_high,pool_type='avg',global_pooling=True)

        print('gap.shape------------', gap.shape)
        h = conv3.shape[2]
        w = conv3.shape[3]
        gap = fluid.layers.resize_bilinear(input = gap,out_shape = [h,w] )

        conv1conv3 = fluid.layers.elementwise_mul(gap, conv3)
        '''
        conv1 = conv_bn_layer(input=gap, num_filters=3*channels,
        filter_size=1, stride=1,dilation=1, act='relu')
        print("conv1.shape---------",conv1.shape)
        '''
        #out = fluid.layers.sequence_concat(input=[conv1conv3, inputs_high])
        out = fluid.layers.concat([conv1conv3, inputs_high], axis = 1)

        return out

    def model(self, inputs):

        conv1 = self.Inception_dilation(inputs, 4)
        res1 = fluid.layers.concat([inputs, conv1], axis = 1)
        conv2 = self.Inception_dilation(res1, 4)
        conv2 = self.Inception_dilation(conv2, 4)
        res2 = fluid.layers.concat([res1, conv2], axis = 1)
        conv3 = self.Inception_dilation(res2, 4)
        conv3 = self.Inception_dilation(conv3, 4)
        res3 = fluid.layers.concat([res2, conv3], axis = 1)
        conv4 = self.Inception_dilation(res3, 4)
        conv4 = self.Inception_dilation(conv4, 4)

        FPA = self.FeaturePyramidAttention(conv4, 4)
        print('FPA.shape', FPA.shape)
        print('conv3.shape', conv3.shape)
        GAU1 = self.GlobalAttentionUpsample(conv3, FPA, 4)
        GF1 = fluid.layers.concat([FPA, GAU1], axis = 1)

        GAU2 = self.GlobalAttentionUpsample(conv2, GF1, 12)
        GF2 = fluid.layers.concat([GF1, GAU2], axis = 1)

        GAU3 = self.GlobalAttentionUpsample(conv1, GF2, 36)
        GF3 = fluid.layers.concat([GF2, GAU3], axis = 1)

        conv8 = conv_bn_layer(input=GF3, num_filters=12,filter_size=1,
        stride=1,dilation=1, act='relu')
        print("conv8 shape:", conv8.shape)

        conv9 = conv_bn_layer(input=conv8, num_filters=9,filter_size=1,
        stride=1,dilation=1, act='relu')

        conv9 = fluid.layers.transpose(x=conv9, perm=[0, 2, 3, 1])
        conv9 = fluid.layers.reshape(conv9, shape=[-1, 9])
        modelOut = fluid.layers.softmax(conv9)
        print('modelOut.shape == ',modelOut.shape)

        return modelOut