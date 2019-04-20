#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

from models.factory import build_net
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr


parser = argparse.ArgumentParser(description='dsfd demo')
parser.add_argument('--network',
                    default='vgg', type=str,
                    choices=['vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--save_dir',
                    type=str, default='tmp/',
                    help='Directory for detect result')
parser.add_argument('--model',
                    type=str,
                    default='weights/dsfd_face.pth', help='trained model')
parser.add_argument('--thresh',
                    default=0.4, type=float,
                    help='Final confidence threshold')
parser.add_argument('--data_dir',
                    type=str, default='/images',
                    help='directory to load data')
parser.add_argument('--res_dir',
                    type=str, default='/predictions',
                    help='directory to save res')
args = parser.parse_args()

if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)
if not os.path.exists(args.data_dir):
    print(args.data_dir, ' not found!')
    exit

print('data_dir', args.data_dir)
print('res_dir', args.res_dir)
# input()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def detect(net, img_path, thresh, imgName):
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1500 * 1000 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(img, None, None, fx=max_im_shrink,
                       fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= np.array([103.939, 116.779, 123.68])[:, np.newaxis, np.newaxis].astype('float32')
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()
    t1 = time.time()
    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])
    # print(detections)
    # print(detections.size())
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    f = open(os.path.join(args.res_dir, imgName.replace('png', 'txt')), "w")
    for i in range(detections.size(1)):
      j = 0
      # while detections[0, i, j, 0] >= thresh:
      while ((j < detections.size(2)) and detections[0, i, j, 0] > thresh):
        pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)
        left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
        score = detections[0, i, j, 0]
        f.write("%f %f %f %f %f\n" % (pt[0], pt[1], pt[2], pt[3], score))
        j += 1
    f.close();
    """
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)
            left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            j += 1
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
            conf = "{:.2f}".format(score)
            text_size, baseline = cv2.getTextSize(
                conf, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            p1 = (left_up[0], left_up[1] - text_size[1])
            cv2.rectangle(img, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
                          (p1[0] + text_size[0], p1[1] + text_size[1]),[255,0,0], -1)
            cv2.putText(img, conf, (p1[0], p1[
                            1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, 8)
    """
    t2 = time.time()
    print('detect:{} timer:{}'.format(img_path, t2 - t1))
    # cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)


if __name__ == '__main__':
    net = build_net('test', 2, args.network)
    # net.load_state_dict(torch.load(args.model, map_location='cuda:3'))
    net.load_state_dict(torch.load('./weights/dsfd_vgg_0.880.pth'))
    # net.load_state_dict(torch.load('/home/xudejia/DSFD.pytorch/weights2/vgg/bak/dsfd_4000.pth'))
    # net.load_state_dict(torch.load('weights/vgg/dsfd_11000.pth'))
    # net.load_state_dict(torch.load(args.model))
    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    
    cur_folder = args.data_dir
    # # img_path = '/mnt/hdd/xudejia/Data/selected'
    # # img_path = '/home/xudejia/Data/'
    # folder_list = [x for x in os.listdir(img_path)]
    # # folder_list.remove('ignore')
    # folder_list = folder_list
    # folder_list = folder_list[5:]
    img_list = [x for x in os.listdir(cur_folder) if x.endswith('png')]
    imgNum = len(img_list)
    print('Start ' + cur_folder)
    # print(imgN)
    doneNum = 0
    for path in img_list:
        # img_path is ~/Data
        # path is xxx.jpg
        # print((img_path, path))
        # path should be like `2/xxx.jpg`
        # so that res can be saved into the right place
        with torch.no_grad():
            detect(net, os.path.join(cur_folder, path), args.thresh, path)
        doneNum += 1
        print('Finished %s, %d / %d\n' % (os.path.join(cur_folder, path), doneNum, imgNum))
