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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class deeplabv3p_ours(object):
    def __init__(self, rows=512, cols=512):
        self.rows = rows
        self.cols = cols

    def xception_downsample_block(self,x,channels,top_relu=False):
        if top_relu:
            x = fluid.layers.relu(x)

        x0_d = depthwise_bn_layer(input=x, filter_size=3,  dilation=1,
        act=None)
        x0_p = conv_bn_layer(input=x0_d, num_filters=channels, filter_size=1,
        act='relu')
        x1_d = depthwise_bn_layer(input=x0_p, filter_size=3,  dilation=1,
        act=None)
        x1_p = conv_bn_layer(input=x1_d, num_filters=channels, filter_size=1,
        act='relu')
        x2_d = depthwise_bn_layer(input=x1_p, filter_size=3, stride=2, dilation=1,
        act=None)

        x2_p = conv_bn_layer(input=x2_d, num_filters=channels, filter_size=1,
        act=None)

        return x2_p

    def res_xception_downsample_block(self,x,channels):
        res = conv_bn_layer(input=x, num_filters=channels,filter_size=1,
        stride=2, act=None)
        x = self.xception_downsample_block(x,channels)
        x = x + res  # add
        '''
        "x=add([x,res])" means just add, not concate
        source link:https://blog.csdn.net/u012193416/article/details/79479935
        '''
        return x

    def xception_block(self,x,channels):
        x = fluid.layers.relu(x)
        x0_d = depthwise_bn_layer(input=x, filter_size=3,  dilation=1,
        act='relu')
        x0_p = conv_bn_layer(input=x0_d, num_filters=channels, filter_size=1,
        act='relu')
        x1_d = depthwise_bn_layer(input=x0_p, filter_size=3,  dilation=1,
        act='relu')
        x1_p = conv_bn_layer(input=x1_d, num_filters=channels, filter_size=1,
        act='relu')

        return x1_p

    def res_xception_block(self,x,channels):
        res = x
        x = self.xception_block(x,channels)
        x = x + res
        return x



    def assp(self,x,input_shape,out_stride):
        b0 = conv_layer(
        input=x, num_filters=256, filter_size=1,  act=None)
        b1 = depthwise_layer(input=x, filter_size=3, dilation=6, act=None)
        b1 = fluid.layers.batch_norm(input=b1)
        b1 = fluid.layers.relu(b1)
        b1 = conv_bn_layer(input=b1, num_filters=256, filter_size=1, act="relu")

        b2 = depthwise_layer(input=x, filter_size=3, dilation=12, act=None)
        b2 = fluid.layers.batch_norm(input=b2)
        b2 = fluid.layers.relu(b2)
        b2 = conv_bn_layer(input=b2, num_filters=256, filter_size=1, act="relu")

        b3 = depthwise_layer(input=x, filter_size=3, dilation=12, act=None)
        b3 = fluid.layers.batch_norm(input=b3)
        b3 = fluid.layers.relu(b3)
        b3 = conv_bn_layer(input=b3, num_filters=256, filter_size=1, act="relu")
        in_shape = input_shape[1] if input_shape[1] < input_shape[2] else input_shape[2]
        out = int(in_shape/out_stride)
        b4 = fluid.layers.pool2d(input=x, pool_size=out, pool_stride=out,
        pool_type='avg')
        b4 = conv_bn_layer(input=b4, num_filters=256, filter_size=1, act="relu")
        b4 = fluid.layers.resize_bilinear(input = b4,scale = out)
        x = fluid.layers.concat([b4,b0,b1,b2,b3],axis=1)
        return x


    def model(self,input,input_shape=(3,1024,1024),out_stride=16):

        x0 = conv_bn_layer(
        input=input, num_filters=32, filter_size=3, stride=2, act='relu')

        x1 = conv_bn_layer(
        input=x0, num_filters=64, filter_size=3,  act='relu')

        x1r = fluid.layers.resize_bilinear(input = x1,scale= 0.5)


        x2 = self.res_xception_downsample_block(x1,128)


        res = conv_bn_layer(
        input=x2, num_filters=256, filter_size=1, stride=2, act='relu')


        x = depthwise_bn_layer(input=x2, filter_size=3,  act=None)

        x = conv_bn_layer(
        input=x, num_filters=256, filter_size=1,  act='relu')

        x = depthwise_bn_layer(input=x, filter_size=3,  act=None)

        skip = conv_bn_layer(
        input=x, num_filters=256, filter_size=1,  act=None)

        x = fluid.layers.relu(skip)

        x = depthwise_bn_layer(input=x, filter_size=3, stride=2, act=None)

        x3 = conv_bn_layer(
        input=x, num_filters=256, filter_size=1,  act=None)


        x3r = fluid.layers.resize_bilinear(input = x3,scale= 2)

        x = x3 + res

        x4 = depthwise_bn_layer(input=x, filter_size=3, stride=2, act=None)

        x5 = conv_bn_layer(
        input=x4, num_filters=256, filter_size=1,  act=None)


        x = self.xception_downsample_block(x,728,top_relu=True)
        for i in range(16):
            x = self.res_xception_block(x,728)

        res = conv_bn_layer(input=x, num_filters=1024, filter_size=1,  act=None)

        x = fluid.layers.relu(x)
        x = depthwise_bn_layer(input=x, filter_size=3, act=None)
        x = conv_bn_layer(input=x, num_filters=728, filter_size=1, act="relu")
        x = depthwise_bn_layer(input=x, filter_size=3, act=None)
        x = conv_bn_layer(input=x, num_filters=1024, filter_size=1, act="relu")
        x = depthwise_bn_layer(input=x, filter_size=3, act=None)
        x = conv_bn_layer(input=x, num_filters=1024, filter_size=1, act=None)
        x = x + res
        x = depthwise_bn_layer(input=x, filter_size=3, act=None)
        x = conv_bn_layer(input=x, num_filters=1536, filter_size=1, act="relu")
        x = depthwise_bn_layer(input=x, filter_size=3, act=None)
        x = conv_bn_layer(input=x, num_filters=1536, filter_size=1, act="relu")
        x = depthwise_bn_layer(input=x, filter_size=3, act=None)
        x = conv_bn_layer(input=x, num_filters=2048, filter_size=1, act="relu")

        #assp
        x = self.assp(x,input_shape,out_stride)
        x = conv_bn_layer(input=x, num_filters=256, filter_size=1, act="relu")
        x = fluid.layers.dropout(x, dropout_prob=0.9)

        #decoder

        x = fluid.layers.concat([x5, x],axis=1)
        x6 = fluid.layers.resize_bilinear(input = x,scale= 4)

        dec_skip = conv_bn_layer(input=skip, num_filters=48, filter_size=1, act="relu")
        x = fluid.layers.concat([x6, dec_skip],axis=1)


        x = depthwise_bn_layer(input=x, filter_size=3, act="relu")
        x7 = conv_bn_layer(input=x, num_filters=256, filter_size=1, act="relu")

        x = fluid.layers.concat([x7, x3r],axis=1)

        x = depthwise_bn_layer(input=x, filter_size=3, act="relu")
        x8 = conv_bn_layer(input=x, num_filters=128, filter_size=1, act="relu")

        x = fluid.layers.concat([x8, x2],axis=1)

        x = depthwise_bn_layer(input=x, filter_size=3, act="relu")
        x9 = conv_bn_layer(input=x, num_filters=64, filter_size=1, act="relu")

        x = fluid.layers.concat([x9, x1r],axis=1)



        x = conv_layer(input=x, num_filters=9, filter_size=1, act=None)

        x = fluid.layers.resize_bilinear(input = x,scale= 4)

        x = fluid.layers.transpose(x=x, perm=[0, 2, 3, 1])
        modelOut = fluid.layers.reshape(x, shape=[-1, 9])
        modelOut = fluid.layers.softmax(modelOut)

        print('model_out',modelOut.shape)
        return modelOut

