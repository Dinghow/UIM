import os
import os.path
import sys
import errno
import hashlib
import tarfile
import json
import cv2
import math
import copy
import random
import scipy.io
import numpy as np
from PIL import Image
from six.moves import urllib
import logging
import glob
import functools
import numbers
import util.helpers as helpers
from util.util import load_image_file

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


class Composition1KMatting(Dataset):
    BASE_DIR = 'Combined_Dataset'

    def __init__(self,
                 root='dataset',
                 split='train',
                 task='matting',
                 num_bgs=10,
                 transform=None,
                 preprocess=False,
                 retname=True):        
        assert task in ['seg', 'matting']
        if 'train' in split:
            bg_name_file = 'training_bg_names.txt'
        if 'test' in split:
            # assert num_bgs == 20 
            bg_name_file = 'test_bg_names.txt'
        self.root = root
        self.split = split
        self.task = task
        self.num_bgs = num_bgs
        self.split_dir = 'Training_set' if 'train' in split else 'Test_set'
        self.ext = '.jpg' if 'train' in split else '.png'
        self._composition_dir = os.path.join(self.root, self.BASE_DIR, self.split_dir)
        self._alpha_dir = os.path.join(self._composition_dir, 'alpha')
        self._fg_dir = os.path.join(self._composition_dir, 'fg')
        self._bg_dir = os.path.join(self.root, self._composition_dir, 'bg')
        self._split_dir = os.path.join(self._composition_dir, 'ImageSets')
        self._trimap_dir = os.path.join(self._composition_dir, 'trimaps')
        self.transform = transform
        self.retname = retname

        with open(os.path.join(os.path.join(self._split_dir, self.split + '.txt')), "r") as f:
            self.sample_list = f.read().splitlines()

        self.im_ids = []
        self.alphas = []
        self.fgs = []
        self.bgs = []
        self.obj_dict = dict()
        self.fg_num = len(self.sample_list)

        for ii, sample_name in enumerate(self.sample_list):
            for jj in range(num_bgs):
                im_id = sample_name + '_' + str(jj)
                self.im_ids.append(im_id)
                self.alphas.append(sample_name)
                self.fgs.append(sample_name)
                # self.bgs.append(self.bg_list[ii*num_bgs + jj])
                self.bgs.append(im_id)
                self.obj_dict[im_id] = [-2]

        # Build the list of objects
        self.obj_list = []
        num_images = 0

        for ii in range(len(self.im_ids)):
            flag = False
            for jj in range(len(self.obj_dict[self.im_ids[ii]])):
                if self.obj_dict[self.im_ids[ii]][jj] != -1:
                    self.obj_list.append([ii, jj])
                    flag = True
            if flag:
                num_images += 1
        
        self.sample_num = len(self.obj_list)
        # Display stats
        print('Number of images: {:d}\nNumber of objects: {:d}'.format(num_images, len(self.obj_list)))

    def __getitem__(self, index):
        alpha = cv2.imread(os.path.join(self._alpha_dir, self.alphas[index] + self.ext), 0).astype(np.float32) / 255.
        fg = cv2.imread(os.path.join(self._fg_dir, self.fgs[index] + self.ext), 1)
        bg = cv2.imread(os.path.join(self._bg_dir, self.bgs[index] + '.png'), 1)
        if 'train' in self.split:
            fg, alpha = self._composite_fg(fg, alpha, index)
        else:
            trimap = cv2.imread(os.path.join(self._trimap_dir, self.im_ids[index] + self.ext), 0)
            h, w = fg.shape[:2]
            bh, bw = bg.shape[:2]
            wratio = w / bw
            hratio = h / bh
            ratio = wratio if wratio > hratio else hratio
            if ratio > 1:
                bg = cv2.resize(src=bg, dsize=(math.ceil(bw*ratio), math.ceil(bh*ratio)), interpolation=cv2.INTER_CUBIC)
            im, alpha, fg, bg, trimap = self._composite(fg, bg, alpha, w, h, trimap)

        if self.task == 'seg':
            alpha = (alpha > (alpha.max()/2)).astype(np.float32) # Transfer matting alpha to binary mask

        if 'train' in self.split:
            sample = {'alpha': alpha, 'fg': fg, 'bg': bg}
        else:
            sample = {'image': im, 'alpha': alpha, 'fg': fg, 'bg': bg, 'trimap': trimap, 'alpha_shape': alpha.shape, 'trimap_ori': trimap.copy(), 'alpha_ori': alpha.copy()}

        if self.retname:
            _im_ii = self.obj_list[index][0]
            _obj_ii = self.obj_list[index][1]
            _target_area = np.where(alpha>0)
            sample['meta'] = {'image': str(self.im_ids[_im_ii]),
                              'split_dir': str(self._split_dir),
                              'object': str(_obj_ii),
                              'category': 1,
                              'im_size': (alpha.shape[0], alpha.shape[1]),
                              'boundary': [_target_area[0].max(), _target_area[0].min(),
                                           _target_area[1].max(), _target_area[1].min()]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _composite_fg(self, fg, alpha, idx):

        if np.random.rand() < 0.5:
            idx2 = np.random.randint(self.sample_num) + idx
            fg2 = cv2.imread(os.path.join(self._fg_dir, self.fgs[idx2 % self.sample_num] + self.ext), 1)
            alpha2 = cv2.imread(os.path.join(self._alpha_dir, self.alphas[idx2 % self.sample_num] + self.ext), 0).astype(np.float32) / 255.
            h, w = alpha.shape
            fg2 = cv2.resize(fg2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha2 = cv2.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

            alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
            if np.any(alpha_tmp < 1):
                fg = fg.astype(np.float32) * alpha[:, :, None] + fg2.astype(np.float32) * (1 - alpha[:, :, None])
                # The overlap of two 50% transparency should be 25%
                alpha = alpha_tmp
                fg = fg.astype(np.uint8)

        if np.random.rand() < 0.25:
            fg = cv2.resize(fg, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha = cv2.resize(alpha, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        return fg, alpha

    def _composite(self, fg, bg, a, w, h, trimap):
        fg = np.array(fg, np.float32)
        bg_h, bg_w = bg.shape[:2]
        x = max(0, int((bg_w - w) / 2))
        y = max(0, int((bg_h - h) / 2))
        crop = np.array(bg[y:y + h, x:x + w], np.float32)
        alpha = np.zeros((h, w, 1), np.float32)
        alpha[:, :, 0] = a
        im = alpha * fg + (1 - alpha) * crop
        im = im.astype(np.uint8)

        new_a = np.zeros((bg_h, bg_w), np.float32)
        new_a[y:y + h, x:x + w] = a
        new_trimap = np.zeros((bg_h, bg_w), np.uint8)
        new_trimap[y:y + h, x:x + w] = trimap
        new_im = bg.copy()
        new_im[y:y + h, x:x + w] = im
        return new_im, new_a, fg, bg, new_trimap

    def __len__(self):
        return len(self.obj_list)

    def __str__(self):
        return 'Composition-1K(split=' + str(self.split) + ')'
 

class Combined4ClassesMatting(Dataset):
    def __init__(self,
                 root='dataset',
                 split='all',
                 transform=None,
                 retname=True):        
        assert split in ['all', 'SO', 'STM', 'NSO', 'NSTM']
        self.root = root
        self.split = split
        self.transform = transform
        self.retname = retname
        self.img_dir = os.path.join(self.root, 'images')
        self.alpha_dir = os.path.join(self.root, 'alphas')
        self.trimap_dir = os.path.join(self.root, 'trimaps')
        self.category_dic = {'SO': 0, 'STM': 1, 'NSO': 2, 'NSTM': 3}

        self.im_ids = []
        self.alphas = []
        self.trimaps = []
        self.images = []

        data_types = ['SO', 'STM', 'NSO', 'NSTM']
        for data_type in data_types:
            if self.split == data_type or self.split == 'all':
                img_cat_dir = os.path.join(self.img_dir, data_type)
                filenames = os.listdir(img_cat_dir)
                img_list = []
                for filename in filenames:
                    if '.png' in filename or '.jpg' in filename:
                        img_list.append(filename)
                    # if '.jpg' in filename:
                    #     img_list.append(filename)
                img_list.sort()
                for img_name in img_list:
                    sample_name = img_name.split('.')[0]
                    self.im_ids.append(sample_name)
                    self.images.append(os.path.join(img_cat_dir, img_name))
                    self.alphas.append(os.path.join(self.alpha_dir, sample_name + '.png'))
                    self.trimaps.append(os.path.join(self.trimap_dir, sample_name + '.png'))
            else:
                continue
        # Display stats
        print('Number of images: {:d}'.format(len(self.im_ids)))

    def __getitem__(self, index):
        alpha = cv2.imread(self.alphas[index], 0).astype(np.float32) / 255.
        image = cv2.imread(self.images[index], 1)
        trimap = cv2.imread(self.trimaps[index], 0)

        sample = {'image': image, 'alpha': alpha, 'trimap': trimap, 'alpha_shape': alpha.shape, 'trimap_ori': trimap.copy(), 'alpha_ori': alpha.copy()}

        if self.retname:
            _target_area = np.where(alpha>0)
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'category': self.category_dic[self.im_ids[index].split('_')[0]],
                              'im_size': (alpha.shape[0], alpha.shape[1]),
                              'boundary': [_target_area[0].max(), _target_area[0].min(),
                                           _target_area[1].max(), _target_area[1].min()]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.im_ids)

    def __str__(self):
        return 'Combined-4classes-Test set'


class RealWorldMatting(Dataset):
    def __init__(self,
                 root='dataset',
                 transform=None,
                 retname=True):        
        self.root = root
        self.transform = transform
        self.retname = retname
        self.img_dir = os.path.join(self.root, 'images_new')
        self.mask_dir = os.path.join(self.root, 'masks_new')

        self.im_ids = []
        self.masks = []
        self.images = []

        filenames = os.listdir(self.img_dir)
        img_list = []
        for filename in filenames:
            if '.png' in filename or '.jpg' in filename:
                img_list.append(filename)
        img_list.sort()
        for img_name in img_list:
            sample_name = img_name.split('.')[0]
            self.im_ids.append(sample_name)
            self.images.append(os.path.join(self.img_dir, img_name))
            self.masks.append(os.path.join(self.mask_dir, sample_name + '.png'))
        # Display stats
        print('Number of images: {:d}'.format(len(self.im_ids)))

    def __getitem__(self, index):
        mask = cv2.imread(self.masks[index], 0).astype(np.float32) / 255.
        image = cv2.imread(self.images[index], 1)

        sample = {'image': image, 'alpha': mask, 'trimap': mask * 255, 'alpha_shape': mask.shape, 'trimap_ori': mask.copy(), 'alpha_ori': mask.copy()}

        if self.retname:
            _target_area = np.where(mask>0)
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'im_size': (mask.shape[0], mask.shape[1]),
                              'boundary': [_target_area[0].max(), _target_area[0].min(),
                                           _target_area[1].max(), _target_area[1].min()]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.im_ids)

    def __str__(self):
        return 'Real world test data'