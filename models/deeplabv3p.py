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


class deeplab_v3p(object):
    def __init__(self,):
        self.name_scope = ""
        self.decode_channel = 48
        self.encode_channel = 256
        self.label_number = 9
        self.bn_momentum = 0.99
        self.dropout_keep_prop = 0.9
        self.is_train = True
        self.op_results = {}
        self.default_epsilon = 1e-3
        self.default_norm_type = 'bn'
        self.default_group_number = 32
        self.depthwise_use_cudnn = False
        self.bn_regularizer = fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.0)
        self.depthwise_regularizer = fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.0)
        self.clean()

    @contextlib.contextmanager
    def scope(self,name):
        #global self.name_scope
        bk = self.name_scope
        self.name_scope = self.name_scope + name + '/'
        yield
        self.name_scope = bk

    def check(self,data, number):
        if type(data) == int:
            return [data] * number
        assert len(data) == number
        return data


    def clean(self):
        #global self.op_results
        self.op_results = {}


    def append_op_result(self,result, name):
        #global self.op_results
        op_index = len(self.op_results)
        name = self.name_scope + name + str(op_index)
        self.op_results[name] = result
        return result


    def conv(self,*args, **kargs):
        if "xception" in self.name_scope:
            init_std = 0.09
        elif "logit" in self.name_scope:
            init_std = 0.01
        elif self.name_scope.endswith('depthwise/'):
            init_std = 0.33
        else:
            init_std = 0.06
        if self.name_scope.endswith('depthwise/'):
            regularizer = self.depthwise_regularizer
        else:
            regularizer = None

        kargs['param_attr'] = fluid.ParamAttr(
            name=self.name_scope + 'weights',
            regularizer=regularizer,
            initializer=fluid.initializer.TruncatedNormal(
                loc=0.0, scale=init_std))
        if 'bias_attr' in kargs and kargs['bias_attr']:
            kargs['bias_attr'] = fluid.ParamAttr(
                name=self.name_scope + 'biases',
                regularizer=regularizer,
                initializer=fluid.initializer.ConstantInitializer(value=0.0))
        else:
            kargs['bias_attr'] = False
        kargs['name'] = self.name_scope + 'conv'
        return self.append_op_result(fluid.layers.conv2d(*args, **kargs), 'conv')


    def group_norm(self,input, G, eps=1e-5, param_attr=None, bias_attr=None):
        N, C, H, W = input.shape
        if C % G != 0:
            # print "group can not divide channle:", C, G
            for d in range(10):
                for t in [d, -d]:
                    if G + t <= 0: continue
                    if C % (G + t) == 0:
                        G = G + t
                        break
                if C % G == 0:
                    # print "use group size:", G
                    break
        assert C % G == 0
        x = fluid.layers.group_norm(
            input,
            groups=G,
            param_attr=param_attr,
            bias_attr=bias_attr,
            name=self.name_scope + 'group_norm')
        return x


    def bn(self,*args, **kargs):
        if self.default_norm_type == 'bn':
            with self.scope('BatchNorm'):
                return self.append_op_result(
                    fluid.layers.batch_norm(
                        *args,
                        epsilon=self.default_epsilon,
                        momentum=self.bn_momentum,
                        param_attr=fluid.ParamAttr(
                            name=self.name_scope + 'gamma', regularizer=self.bn_regularizer),
                        bias_attr=fluid.ParamAttr(
                            name=self.name_scope + 'beta', regularizer=self.bn_regularizer),
                        moving_mean_name=self.name_scope + 'moving_mean',
                        moving_variance_name=self.name_scope + 'moving_variance',
                        **kargs),
                    'bn')
        elif self.default_norm_type == 'gn':
            with self.scope('GroupNorm'):
                return self.append_op_result(
                    self.group_norm(
                        args[0],
                        self.default_group_number,
                        eps=self.default_epsilon,
                        param_attr=fluid.ParamAttr(
                            name=self.name_scope + 'gamma', regularizer=self.bn_regularizer),
                        bias_attr=fluid.ParamAttr(
                            name=self.name_scope + 'beta', regularizer=self.bn_regularizer)),
                    'gn')
        else:
            raise "Unsupport norm type:" + self.default_norm_type


    def bn_relu(self,data):
        return self.append_op_result(fluid.layers.relu(self.bn(data)), 'relu')


    def relu(self,data):
        return self.append_op_result(fluid.layers.relu(data), 'relu')


    def seq_conv(self,input, channel, stride, filter, dilation=1, act=None):
        with self.scope('depthwise'):
            input = self.conv(
                input,
                input.shape[1],
                filter,
                stride,
                groups=input.shape[1],
                padding=(filter // 2) * dilation,
                dilation=dilation,
                use_cudnn=self.depthwise_use_cudnn)
            input = self.bn(input)
            if act: input = act(input)
        with self.scope('pointwise'):
            input = self.conv(input, channel, 1, 1, groups=1, padding=0)
            input = self.bn(input)
            if act: input = act(input)
        return input


    def xception_block(self,input,
                    channels,
                    strides=1,
                    filters=3,
                    dilation=1,
                    skip_conv=True,
                    has_skip=True,
                    activation_fn_in_separable_conv=False):
        repeat_number = 3
        channels = self.check(channels, repeat_number)
        filters = self.check(filters, repeat_number)
        strides = self.check(strides, repeat_number)
        data = input
        results = []
        for i in range(repeat_number):
            with self.scope('separable_conv' + str(i + 1)):
                if not activation_fn_in_separable_conv:
                    data = self.relu(data)
                    data = self.seq_conv(
                        data,
                        channels[i],
                        strides[i],
                        filters[i],
                        dilation=dilation)
                else:
                    data = self.seq_conv(
                        data,
                        channels[i],
                        strides[i],
                        filters[i],
                        dilation=dilation,
                        act=self.relu)
                results.append(data)
        if not has_skip:
            return self.append_op_result(data, 'xception_block'), results
        if skip_conv:
            with self.scope('shortcut'):
                skip = self.bn(
                    self.conv(
                        input, channels[-1], 1, strides[-1], groups=1, padding=0))
        else:
            skip = input
        return self.append_op_result(data + skip, 'xception_block'), results


    def entry_flow(self,data):
        with self.scope("entry_flow"):
            with self.scope("conv1"):
                data = self.conv(data, 32, 3, stride=2, padding=1)
                data = self.bn_relu(data)
            with self.scope("conv2"):
                data = self.conv(data, 64, 3, stride=1, padding=1)
                data = self.bn_relu(data)
            with self.scope("block1"):
                data, _ = self.xception_block(data, 128, [1, 1, 2])
            with self.scope("block2"):
                data, results = self.xception_block(data, 256, [1, 1, 2])
            with self.scope("block3"):
                data, _ = self.xception_block(data, 728, [1, 1, 2])
            return data, results[1]


    def middle_flow(self,data):
        with self.scope("middle_flow"):
            for i in range(16):
                with self.scope("block" + str(i + 1)):
                    data, _ = self.xception_block(data, 728, [1, 1, 1], skip_conv=False)
        return data


    def exit_flow(self,data):
        with self.scope("exit_flow"):
            with self.scope('block1'):
                data, _ = self.xception_block(data, [728, 1024, 1024], [1, 1, 1])
            with self.scope('block2'):
                data, _ = self.xception_block(
                    data, [1536, 1536, 2048], [1, 1, 1],
                    dilation=2,
                    has_skip=False,
                    activation_fn_in_separable_conv=True)
            return data


    def dropout(self,x, keep_rate):
        if self.is_train:
            return fluid.layers.dropout(x, 1 - keep_rate) / keep_rate
        else:
            return x


    def encoder(self,input):
        with self.scope('encoder'):
            channel = 256
            with self.scope("image_pool"):
                image_avg = fluid.layers.reduce_mean(input, [2, 3], keep_dim=True)
                self.append_op_result(image_avg, 'reduce_mean')
                image_avg = self.bn_relu(
                    self.conv(
                        image_avg, channel, 1, 1, groups=1, padding=0))
                image_avg = fluid.layers.resize_bilinear(image_avg, input.shape[2:])

            with self.scope("aspp0"):
                aspp0 = self.bn_relu(self.conv(input, channel, 1, 1, groups=1, padding=0))
            with self.scope("aspp1"):
                aspp1 = self.seq_conv(input, channel, 1, 3, dilation=6, act=self.relu)
            with self.scope("aspp2"):
                aspp2 = self.seq_conv(input, channel, 1, 3, dilation=12, act=self.relu)
            with self.scope("aspp3"):
                aspp3 = self.seq_conv(input, channel, 1, 3, dilation=18, act=self.relu)
            with self.scope("concat"):
                data = self.append_op_result(
                    fluid.layers.concat(
                        [image_avg, aspp0, aspp1, aspp2, aspp3], axis=1),
                    'concat')
                data = self.bn_relu(self.conv(data, channel, 1, 1, groups=1, padding=0))
                data = self.dropout(data, self.dropout_keep_prop)
            return data


    def decoder(self,encode_data, decode_shortcut):
        with self.scope('decoder'):
            with self.scope('concat'):
                decode_shortcut = self.bn_relu(
                    self.conv(
                        decode_shortcut, self.decode_channel, 1, 1, groups=1, padding=0))
                encode_data = fluid.layers.resize_bilinear(
                    encode_data, decode_shortcut.shape[2:])
                encode_data = fluid.layers.concat(
                    [encode_data, decode_shortcut], axis=1)
                self.append_op_result(encode_data, 'concat')
            with self.scope("separable_conv1"):
                encode_data = self.seq_conv(
                    encode_data, self.encode_channel, 1, 3, dilation=1, act=self.relu)
            with self.scope("separable_conv2"):
                encode_data = self.seq_conv(
                    encode_data, self.encode_channel, 1, 3, dilation=1, act=self.relu)
            return encode_data


    def model(self,img):
        #global self.default_epsilon
        self.append_op_result(img, 'img')
        print('img = ',img.shape)
        with self.scope('xception_65'):
            self.default_epsilon = 1e-3
            # Entry flow
            data, decode_shortcut = self.entry_flow(img)
            # Middle flow
            data = self.middle_flow(data)
            # Exit flow
            data = self.exit_flow(data)
        self.default_epsilon = 1e-5
        encode_data = self.encoder(data)
        encode_data = self.decoder(encode_data, decode_shortcut)
        with self.scope('logit'):
            logit = self.conv(
                encode_data, self.label_number, 1, stride=1, padding=0, bias_attr=True)
            logit = fluid.layers.resize_bilinear(logit, img.shape[2:])
            logit = fluid.layers.transpose(x=logit, perm=[0, 2, 3, 1])
            print('v3p_model = ',logit.shape)
            logit = fluid.layers.reshape(logit, shape=[-1, 9])
            logit = fluid.layers.softmax(logit)
        return logit
