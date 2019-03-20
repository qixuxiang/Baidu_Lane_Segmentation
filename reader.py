#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import sys
import glob
import cv2
import six
import gc
import paddle
import paddle.fluid as fluid
from collections import namedtuple
import paddle.dataset as dataset

from data_augmentor import DataAugmentor
import data_augmentor

# 路径相关
RootPath = os.path.abspath("./")
sys.path.append(RootPath)
print ("项目根目录路径为: ",RootPath)

# 标注数据类别
Label = namedtuple( 'Label' , [

    'name'        ,
    'id'          ,
    'trainId'     ,
    'category'    ,
    'categoryId'  ,
    'hasInstances',
    'ignoreInEval',
    'color'       ,
    ] )

# 标注定义
labels = [
    #           name     id trainId      category  catId hasInstances ignoreInEval            color
    Label(     'void' ,   0 ,     0,        'void' ,   0 ,      False ,      False , (  0,   0,   0) ),
    Label(    's_w_d' , 200 ,     1 ,   'dividing' ,   1 ,      False ,      False , ( 70, 130, 180) ),
    Label(    's_y_d' , 204 ,     1 ,   'dividing' ,   1 ,      False ,      False , (220,  20,  60) ),
    Label(  'ds_w_dn' , 213 ,     1 ,   'dividing' ,   1 ,      False ,       True , (128,   0, 128) ),
    Label(  'ds_y_dn' , 209 ,     1 ,   'dividing' ,   1 ,      False ,      False , (255, 0,   0) ),
    Label(  'sb_w_do' , 206 ,     1 ,   'dividing' ,   1 ,      False ,       True , (  0,   0,  60) ),
    Label(  'sb_y_do' , 207 ,     1 ,   'dividing' ,   1 ,      False ,       True , (  0,  60, 100) ),
    Label(    'b_w_g' , 201 ,     2 ,    'guiding' ,   2 ,      False ,      False , (  0,   0, 142) ),
    Label(    'b_y_g' , 203 ,     2 ,    'guiding' ,   2 ,      False ,      False , (119,  11,  32) ),
    Label(   'db_w_g' , 211 ,     2 ,    'guiding' ,   2 ,      False ,       True , (244,  35, 232) ),
    Label(   'db_y_g' , 208 ,     2 ,    'guiding' ,   2 ,      False ,       True , (  0,   0, 160) ),
    Label(   'db_w_s' , 216 ,     3 ,   'stopping' ,   3 ,      False ,       True , (153, 153, 153) ),
    Label(    's_w_s' , 217 ,     3 ,   'stopping' ,   3 ,      False ,      False , (220, 220,   0) ),
    Label(   'ds_w_s' , 215 ,     3 ,   'stopping' ,   3 ,      False ,       True , (250, 170,  30) ),
    Label(    's_w_c' , 218 ,     4 ,    'chevron' ,   4 ,      False ,       True , (102, 102, 156) ),
    Label(    's_y_c' , 219 ,     4 ,    'chevron' ,   4 ,      False ,       True , (128,   0,   0) ),
    Label(    's_w_p' , 210 ,     5 ,    'parking' ,   5 ,      False ,      False , (128,  64, 128) ),
    Label(    's_n_p' , 232 ,     5 ,    'parking' ,   5 ,      False ,       True , (238, 232, 170) ),
    Label(   'c_wy_z' , 214 ,     6 ,      'zebra' ,   6 ,      False ,      False , (190, 153, 153) ),
    Label(    'a_w_u' , 202 ,     7 ,  'thru/turn' ,   7 ,      False ,       True , (  0,   0, 230) ),
    Label(    'a_w_t' , 220 ,     7 ,  'thru/turn' ,   7 ,      False ,      False , (128, 128,   0) ),
    Label(   'a_w_tl' , 221 ,     7 ,  'thru/turn' ,   7 ,      False ,      False , (128,  78, 160) ),
    Label(   'a_w_tr' , 222 ,     7 ,  'thru/turn' ,   7 ,      False ,      False , (150, 100, 100) ),
    Label(  'a_w_tlr' , 231 ,     7 ,  'thru/turn' ,   7 ,      False ,       True , (255, 165,   0) ),
    Label(    'a_w_l' , 224 ,     7 ,  'thru/turn' ,   7 ,      False ,      False , (180, 165, 180) ),
    Label(    'a_w_r' , 225 ,     7 ,  'thru/turn' ,   7 ,      False ,      False , (107, 142,  35) ),
    Label(   'a_w_lr' , 226 ,     7 ,  'thru/turn' ,   7 ,      False ,      False , (201, 255, 229) ),
    Label(   'a_n_lu' , 230 ,     7 ,  'thru/turn' ,   7 ,      False ,       True , (0,   191, 255) ),
    Label(   'a_w_tu' , 228 ,     7 ,  'thru/turn' ,   7 ,      False ,       True , ( 51, 255,  51) ),
    Label(    'a_w_m' , 229 ,     7 ,  'thru/turn' ,   7 ,      False ,       True , (250, 128, 114) ),
    Label(    'a_y_t' , 233 ,     7 ,  'thru/turn' ,   7 ,      False ,       True , (127, 255,   0) ),
    Label(   'b_n_sr' , 205 ,     8 ,  'reduction' ,   8 ,      False ,      False , (255, 128,   0) ),
    Label(  'd_wy_za' , 212 ,     8 ,  'attention' ,   8 ,      False ,       True , (  0, 255, 255) ),
    Label(  'r_wy_np' , 227 ,     8 , 'no parking' ,   8 ,      False ,      False , (178, 132, 190) ),
    Label( 'vom_wy_n' , 223 ,     8 ,     'others' ,   8 ,      False ,       True , (128, 128,  64) ),
    Label(   'om_n_n' , 250 ,     8 ,     'others' ,   8 ,      False ,      False , (102,   0, 204) ),
    Label(    'noise' , 249 ,     0 ,    'ignored' ,   0 ,      False ,       True , (  0, 153, 153) ),
    Label(  'ignored' , 255 ,     0 ,    'ignored' ,   0 ,      False ,       True , (255, 255, 255) ),
]

# 名字转标注
name2label      = { label.name    : label for label in labels           }

# id转标注
id2label        = { label.id      : label for label in labels           }

# 训练id转标注
trainId2label   = { label.trainId : label for label in reversed(labels) }

print ("标准转换检测 200 --->  1： ",id2label[200].trainId)

# 数据预处理
augmentor = DataAugmentor()
class TrainDataReader:
    def __init__(self, dataset_dir, subset='train',rows=2000, cols=1354, shuffle=True, birdeye=True):
        label_dirname = dataset_dir + subset
        print (label_dirname)

        if six.PY2:
            import commands
            label_files = commands.getoutput(
                "find %s -type f | grep _bin.png | sort" %
                label_dirname).splitlines()
        else:
            import subprocess
            label_files = subprocess.getstatusoutput(
                "find %s -type f | grep _bin.png | sort" %
                label_dirname)[-1].splitlines()

        print ('---')
        print (label_files[0])
        self.label_files = label_files
        self.label_dirname = label_dirname
        self.rows = rows
        self.cols = cols
        self.index = 0
        self.subset = subset
        self.dataset_dir = dataset_dir
        self.shuffle = shuffle
        self.M = 0
        self.Minv = 0
        self.reset()
        self.get_M_Minv()
        self.augmentor = 0
        self.birdeye = birdeye
        print("images total number", len(label_files))

    # 标签转分类 255 ignore ?
    def label2classes(self, label,row,col):
        x = np.zeros([row,col,9]).astype(np.int64)
        for i in range(row):
            for j in range(col):
                    try:
                        trainId = id2label[int(label[i][j])].trainId
                        x[i, j ,trainId] = 1 # 属于第m类，第三维m处值为1
                    except Exception as err:
                        #print('像素级标签值异常',err)
                        pass
        return x


    def get_M_Minv(self):
        # 左上、右上、左下、右下
        src = np.float32([[800, 730], [2583, 730], [0, 1709], [3383, 1709]])
        dst = np.float32([[0, 0], [3999,0], [1300, 3999], [2700, 3999]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst,src)

    def reset(self, shuffle=False):
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.label_files)

    def next_img(self):
        self.index += 1
        if self.index >= len(self.label_files):
            self.reset()

    def prev_img(self):
        if self.index >= 1:
            self.index -= 1

    def get_img(self):
        #if self.augmentor != 0 and self.augmentor < 2:
        #    self.prev_img()
        while True:
            label_name = self.label_files[self.index]
            img_name = label_name.replace('_bin.png', '.jpg')
            img_name = img_name.replace('Label', 'ColorImage')
            label = cv2.imread(label_name,cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(img_name)
            if img is None:
                print("load img failed:", img_name)
                self.next_img()
            else:
                break
        try:
            if self.birdeye ==True:
                warped_img = cv2.warpPerspective(img, self.M, (4000, 4000),flags=cv2.INTER_CUBIC)
                warped_label = cv2.warpPerspective(label, self.M, (4000, 4000),flags=cv2.INTER_NEAREST)
                label = cv2.resize(warped_label, (self.cols, self.rows), interpolation=cv2.INTER_NEAREST)
                img   = cv2.resize(warped_img, (self.cols, self.rows), interpolation=cv2.INTER_CUBIC)
            else:
                label = cv2.resize(label, (self.cols, self.rows), interpolation=cv2.INTER_NEAREST)
                img   = cv2.resize(img, (self.cols, self.rows), interpolation=cv2.INTER_CUBIC)
        except Exception as err:
            print('warped_error: ',err)
            img = np.zeros([self.cols,self.rows,3]).astype(np.uint8)
            label = np.zeros([self.cols,self.rows]).astype(np.uint8)
        # 数据增广
        if self.augmentor != 0:
            if self.augmentor < 2:
                img,label = augmentor.disturb(img, label)
            else :
                self.augmentor = 0
        img   = img.transpose((2,0,1))
        label = self.label2classes(label,self.rows, self.cols) # 转换为 9 分类
        return img, label, label_name

    def get_batch(self, batch_size=1):
        imgs = []
        labels = []
        names = []
        while len(imgs) < batch_size:
            img, label, label_name = self.get_img()
            imgs.append(img)
            labels.append(label)
            names.append(label_name)
            self.next_img()
            self.augmentor += 1
        return np.array(imgs), np.array(labels), names

    def get_batch_generator(self, batch_size, total_step):
        def do_get_batch():
            for i in range(total_step):
                gc.collect() 
                try:
                    imgs, labels, names = self.get_batch(batch_size)
                except Exception as err:
                    imgs, labels, names = self.get_batch(batch_size)
                    print('Generator　异常',err)

                imgs   = imgs.astype(np.float32)
                labels = labels.astype(np.float32)
                imgs  /= 255
                yield i, imgs, labels, names

        batches = do_get_batch()
        try:
            from prefetch_generator import BackgroundGenerator
            batches = BackgroundGenerator(batches, 10)
        except:
            print(
                "You can install 'prefetch_generator' for acceleration of data reading."
            )
        return batches

class TestDataReader:
    def __init__(self, dataset_dir, subset='test',rows=880, cols=596, shuffle=False, birdeye=True):
        image_dirname = os.path.join(dataset_dir,subset)
        print (image_dirname)
        image_files = sorted(glob.glob(image_dirname+"/image/*."+"jpg"))
        print ('---')
        print (image_files[0])
        self.image_files = image_files
        self.image_dirname = image_dirname
        self.rows = rows
        self.cols = cols
        self.index = 0
        self.subset = subset
        self.dataset_dir = dataset_dir
        self.shuffle = shuffle
        self.M = 0
        self.Minv = 0
        self.reset()
        self.get_M_Minv()
        self.birdeye = birdeye
        print("images total number", len(image_files))

    def get_M_Minv(self):
        # 左上、右上、左下、右下
        src = np.float32([[800, 730], [2583, 730], [0, 1709], [3383, 1709]])
        dst = np.float32([[0, 0], [3999,0], [1300, 3999], [2700, 3999]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst,src)

    def reset(self, shuffle=False):
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.image_files)

    def next_img(self):
        self.index += 1
        if self.index >= len(self.image_files):
            self.reset()

    def get_img(self):
        while True:
            img_name = self.image_files[self.index]
            label_name = img_name.replace('.jpg', '.png')
            img = cv2.imread(img_name)
            if img is None:
                print("load img failed:", img_name)
                self.next_img()
            else:
                break
        if self.birdeye == True:
            warped_img = cv2.warpPerspective(img, self.M, (4000, 4000),flags=cv2.INTER_CUBIC)
            img   = cv2.resize(warped_img, (self.cols, self.rows), interpolation=cv2.INTER_CUBIC)
        else:
            img   = cv2.resize(img, (self.cols, self.rows), interpolation=cv2.INTER_CUBIC)
        img   = img.transpose((2,0,1))
        return img, label_name

    def get_batch(self, batch_size=1):
        imgs = []
        labels = []
        names = []
        while len(imgs) < batch_size:
            img, label_name = self.get_img()
            imgs.append(img)
            names.append(label_name)
            self.next_img()
        return np.array(imgs), names

    def get_batch_generator(self, batch_size, total_step):
        def do_get_batch():
            for i in range(total_step):
                imgs   = []
                names  = []
                try:
                    imgs, names = self.get_batch(batch_size)
                except Exception as err:
                    imgs, names = self.get_batch(batch_size)
                    print('Generator　异常',err)
                    
                imgs   = imgs.astype(np.float32)
                imgs  /= 255

                yield i, imgs, names

        batches = do_get_batch()
        try:
            from prefetch_generator import BackgroundGenerator
            batches = BackgroundGenerator(batches,10)
        except:
            print(
                "You can install 'prefetch_generator' for acceleration of data reading."
            )
        return batches

class EvalDataReader:
    def __init__(self, dataset_dir, subset='val',rows=512, cols=1024, shuffle=True, birdeye=True):
        label_dirname = os.path.join(dataset_dir,subset)
        print (label_dirname)
        label_files = sorted(glob.glob(label_dirname+"/label/*."+"png"))
        print ('---')
        print (label_files[0])
        self.label_files = label_files
        self.label_dirname = label_dirname
        self.rows = rows
        self.cols = cols
        self.index = 0
        self.subset = subset
        self.dataset_dir = dataset_dir
        self.shuffle = shuffle
        self.reset()
        self.augmentor = 0
        self.M = 0
        self.Minv = 0
        self.get_M_Minv()
        self.birdeye = birdeye
        print("images total number", len(label_files))

    # 标签转分类 255 ignore ?
    def label2classes(self, label,row,col):
        x = np.zeros([row,col,9]).astype(np.int64)
        for i in range(row):
            for j in range(col):
                    try:
                        trainId = id2label[int(label[i][j])].trainId
                        x[i, j ,trainId] = 1 # 属于第m类，第三维m处值为1
                    except Exception as err:
                        print('像素级标签值异常',err)
                        pass
        return x

    def get_M_Minv(self):
        # 左上、右上、左下、右下
        src = np.float32([[800, 730], [2583, 730], [0, 1709], [3383, 1709]])
        dst = np.float32([[0, 0], [3999,0], [1300, 3999], [2700, 3999]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst,src)

    def reset(self, shuffle=False):
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.label_files)

    def next_img(self):
        self.index += 1
        if self.index >= len(self.label_files):
            self.reset()

    def prev_img(self):
        if self.index >= 1:
            self.index -= 1

    def get_img(self):
        #if self.augmentor != 0 and self.augmentor < 6:
        #    self.prev_img()
        while True:
            label_name = self.label_files[self.index]
            img_name = label_name.replace('label', 'image')
            img_name = img_name.replace('_bin.png', '.jpg')
            label = cv2.imread(label_name,cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(img_name)
            if img is None:
                print("load img failed:", img_name)
                self.next_img()
            else:
                break

        warped_img = cv2.warpPerspective(img, self.M, (4000, 4000),flags=cv2.INTER_CUBIC)
        warped_label = cv2.warpPerspective(label, self.M, (4000, 4000),flags=cv2.INTER_NEAREST)

        if self.birdeye == True:
            label = cv2.resize(warped_label, (self.cols, self.rows), interpolation=cv2.INTER_NEAREST)
            img   = cv2.resize(warped_img, (self.cols, self.rows), interpolation=cv2.INTER_CUBIC)
        else:
            label = cv2.resize(label, (self.cols, self.rows), interpolation=cv2.INTER_NEAREST)
            img   = cv2.resize(img, (self.cols, self.rows), interpolation=cv2.INTER_CUBIC)
            
        img   = img.transpose((2,0,1))
        label = self.label2classes(label,self.rows, self.cols) # 转换为 9 分类
        return img, label, label_name

    def get_batch(self, batch_size=1):
        imgs = []
        labels = []
        names = []
        while len(imgs) < batch_size:
            img, label, label_name = self.get_img()
            imgs.append(img)
            labels.append(label)
            names.append(label_name)
            self.next_img()
            self.augmentor += 1
        return np.array(imgs), np.array(labels), names

    def get_batch_generator(self, batch_size, total_step):
        def do_get_batch():
            for i in range(total_step):
                gc.collect() 
                try:
                    imgs, labels, names = self.get_batch(batch_size)
                except Exception as err:
                    imgs, labels, names = self.get_batch(batch_size)
                    print('Generator　异常',err)

                imgs   = imgs.astype(np.float32)
                labels = labels.astype(np.float32)
                imgs  /= 255

                yield i, imgs, labels, names

        batches = do_get_batch()
        try:
            from prefetch_generator import BackgroundGenerator
            batches = BackgroundGenerator(batches, 10)
        except:
            print(
                "You can install 'prefetch_generator' for acceleration of data reading."
            )
        return batches