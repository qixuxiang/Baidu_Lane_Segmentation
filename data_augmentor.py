#-*- coding:utf-8 -*-
import numpy as np
import cv2
import math
from math import fabs,sin,cos,radians
import random
from random import choice
'''
data agumentor,incude Flip,Rotate,Scale and Translation
can replace the code with Augmentor
source code and docs: https://github.com/mdbloice/Augmentor
'''
path = "/media/airobot/docs/BaiduDatas/apolloscape/apolloscape/train/"
image = "image/170927_063811892_Camera_5.jpg"
label = "label/170927_063811892_Camera_5_bin.png"
flipCode = [1,1]

class DataAugmentor:
    def __init__(self):
        pass
    def random_flip(self, img, code):
        return cv2.flip(img, code)

    def random_rotation(self, img, degree):
        height,width = img.shape[:2]
        heightNew = int(width*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
        widthNew = int(height*fabs(sin(radians(degree)))+width*fabs(cos(radians(degree))))
        matRotation = cv2.getRotationMatrix2D((width/2,height/2),degree,1)
        matRotation[0,2] +=(widthNew-width)/2
        matRotation[1,2] +=(heightNew-height)/2
        imgRotation = cv2.warpAffine(img,matRotation,(widthNew,heightNew))
        return imgRotation

    def rotate(self,image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_NEAREST)
        return rotated

    def tfactor(self,img):
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV);#增加饱和度光照的噪声
        hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2)
        hsv[:,:,1] = hsv[:,:,1]*(0.6+ np.random.random()*0.4)
        hsv[:,:,2] = hsv[:,:,2]*(0.4+ np.random.random()*0.6)
        img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        return img  

    def disturb(self, image, label):
        flip_code = choice(flipCode)
        rotate_degree = random.uniform(160,200)
        scale = random.uniform(1.0,2.0)
        image = self.random_flip(image, flip_code)
        label = self.random_flip(label, flip_code)
        #image = self.rotate(image, rotate_degree,scale=scale)
        #label = self.rotate(label, rotate_degree,scale=scale)
        #image = self.tfactor(image)
        return image, label

if __name__ == '__main__':
    img = cv2.imread(path+image)
    img_label = cv2.imread(path+label)
    img = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_CUBIC)
    img_label = cv2.resize(img_label, (1024, 512), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("origin image",img)
    cv2.imshow("origin label image", img_label)
    augmentor = DataAugmentor()
    image,label = augmentor.disturb(img, img_label)

    cv2.imshow("image",image)
    cv2.imshow("label image", label)
    
    #cv2.imwrite("/home/airobot/1.jpg", image)
    #cv2.imwrite("/home/airobot/2.png", label)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
