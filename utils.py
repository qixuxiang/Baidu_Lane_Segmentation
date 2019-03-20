#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid

import contextlib


def conv_layer(input, num_filters, filter_size, stride=1,
                dilation=1,groups=1,act='relu'):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=dilation*(int((filter_size - 1) / 2)), #1
        dilation=dilation,
        groups=groups,
        act=act,
        param_attr=fluid.initializer.Xavier(uniform=False),
        bias_attr=False)
    return conv

def conv_bn_layer(input, num_filters, filter_size, stride=1,dilation=1,
                groups=1,act='relu'):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=dilation*(int((filter_size - 1) / 2)),
        dilation=dilation,
        groups=groups,
        act=act,
        param_attr=fluid.initializer.Xavier(uniform=False),
        bias_attr=False)
    # param_attr=fluid.initializer.Normal(loc=0.0, scale=2.0),
    outconv = fluid.layers.batch_norm(input=conv)
    return outconv

def deconv_layer(input, num_filters, filter_size=4, stride=2,
                  act='relu'):
    deconv = fluid.layers.conv2d_transpose(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=1,
        act=act,
        param_attr=fluid.initializer.Xavier(uniform=False),
        bias_attr=False)
    return deconv

def deconv_bn_layer(input, num_filters, filter_size=4, stride=2,
                  act='relu'):
    deconv = fluid.layers.conv2d_transpose(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=1,
        act=act,
        param_attr=fluid.initializer.Xavier(uniform=False),
        bias_attr=False)
    outconv = fluid.layers.batch_norm(input=deconv)
    return outconv

def depthwise_layer(input, filter_size=3, stride=1, padding = 1, dilation=1,
                    act='relu'):
    depthwise = fluid.layers.conv2d(
        input=input,
        num_filters=input.shape[1],
        filter_size=filter_size,
        stride=stride,
        padding=dilation*(int((filter_size - 1) / 2)),
        dilation=dilation,
        groups=input.shape[1],
        act=act,
        param_attr=fluid.initializer.Xavier(uniform=False),
        bias_attr=False)

    return depthwise


def depthwise_bn_layer(input, filter_size=3, stride=1, padding = 1, dilation=1,
                    act='relu'):
    depthwise = fluid.layers.conv2d(
        input=input,
        num_filters=input.shape[1],
        filter_size=filter_size,
        stride=stride,
        padding=dilation*(int((filter_size - 1) / 2)),
        dilation=dilation,
        groups=input.shape[1],              #implement depthwise by set groups
        act=act,
        param_attr=fluid.initializer.Xavier(uniform=False),
        bias_attr=False)
    outconv = fluid.layers.batch_norm(input=depthwise)

    return outconv
