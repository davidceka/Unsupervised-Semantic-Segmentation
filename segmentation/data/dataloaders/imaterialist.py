#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import sys
import errno
import cv2
import hashlib
import glob
import tarfile
from matplotlib import image

import numpy as np
import torch.utils.data as data
import torch
from PIL import Image

from data.util.mypath import Path

class iMaterialist(data.Dataset):


    #iMATERIALIST_CATEGORY_NAMES = ['background','shirt, blouse',
    #                      'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest',
    #                      'pants', 'shorts', 'skirt', 'coat', 'dress',
    #                      'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory',
    #                      'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings' , 
    #                      'sock','shoe','bag, wallet','scarf','umbrella','hood']

    iMATERIALIST_CATEGORY_NAMES = ['background',
                                   'shirt, blouse',
                                   'top, t-shirt, sweatshirt',
                                   'sweater',
                                   'cardigan',
                                   'jacket',
                                   'vest',
                                   'pants',
                                   'shorts',
                                   'skirt',
                                   'coat',
                                   'dress',
                                   'jumpsuit',
                                   'cape',
                                   'glasses',
                                   'hat',
                                   'headband, head covering, hair accessory',
                                   'tie',
                                   'glove',
                                   'watch',
                                   'belt',
                                   'leg warmer',
                                   'tights, stockings',
                                   'sock',
                                   'shoe',
                                   'bag, wallet',
                                   'scarf',
                                   'umbrella']
    
    
    def __init__(self, root=Path.db_root_dir('imaterialist-fashion-2020-fgvc7'),
                 split='val', transform=None, ignore_classes=[]):
        # Set paths
        self.root = root
        valid_splits = ['trainaug', 'train', 'val']
        assert(split in valid_splits)
        self.split = split
         
        if split == 'trainaug':
            _semseg_dir = os.path.join(self.root, 'SegmentationClassAug')
        else:
            _semseg_dir = os.path.join(self.root, 'SegmentationClass')

        _image_dir = os.path.join(self.root, 'images')

        # Transform
        self.transform = transform
        # Splits are pre-cut
        print("Initializing dataloader for iMaterialist {} set".format(''.join(self.split)))
        split_file = os.path.join(self.root, 'sets', self.split + '.txt')
        self.images = []
        self.semsegs = []
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images
            _image = os.path.join(_image_dir, line + ".jpg")
            assert os.path.isfile(_image)
            self.images.append(_image)

            # Semantic Segmentation
            _semseg = os.path.join(_semseg_dir, line + '.png')
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

        assert(len(self.images) == len(self.semsegs))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

        
        # List of classes which are remapped to ignore index.
        # This option is used for comparing with other works that consider only a subset of the pascal classes.
        self.ignore_classes = [self.iMATERIALIST_CATEGORY_NAMES.index(class_name) for class_name in ignore_classes]
        #if(len(self.ignore_classes)==0):
        #   self.ignore_classes=list(range(28,46))
    def __getitem__(self, index):
        sample = {}

        # Load image
        _img = self._load_img(index)
        sample['image'] = _img

        # Load pixel-level annotations
        _semseg = self._load_semseg(index)
        if _semseg.shape != _img.shape[:2]:
            _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        sample['semseg'] = _semseg
	
        sample['meta'] = {'im_size': (_img.shape[0], _img.shape[1]),
                          'image_file': self.images[index],
                          'image': os.path.basename(self.semsegs[index]).split('.')[0]}
            
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        w, h = _img.size
        basesize = 500
        if(w>h):
            wpercent = (basesize/float(w))
            hsize = int((float(h)*float(wpercent)))
            _img = _img.resize((basesize,hsize), Image.ANTIALIAS)
        else:
            hpercent=(basesize/float(h))
            wsize=int((float(w)*float(hpercent)))
            _img = _img.resize((wsize,basesize), Image.ANTIALIAS)

        
        return np.array(_img)
        
        # _img = _img.resize((int(w / 7.26), int(h / 7.26)))
        

    def _load_semseg(self, index):
        _image=Image.open(self.semsegs[index])
        w, h = _image.size
        basesize = 500
        if(w>h):
            wpercent = (basesize/float(w))
            hsize = int((float(h)*float(wpercent)))
            _sal = _image.resize((basesize,hsize), Image.ANTIALIAS)
            _semseg = np.array(_sal)
        else:
            hpercent=(basesize/float(h))
            wsize=int((float(w)*float(hpercent)))
            _sal = _image.resize((wsize,basesize), Image.ANTIALIAS)
            _semseg = np.array(_sal)

        
        

        for ignore_class in self.ignore_classes:
            _semseg[_semseg == ignore_class] = 255
        return _semseg

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.root, 'JPEGImages', self.images[idx] + '.jpg'))
        return list(reversed(img.size))

    def __str__(self):
        return 'imaterialist-fashion-2020-fgvc7(split=' + str(self.split) + ')' 

    def get_class_names(self):
        return self.iMATERIALIST_CATEGORY_NAMES

    
if __name__ == '__main__':
    """ For purpose of debugging """
    from matplotlib import pyplot as plt
    dataset = iMaterialist(split='train', transform=None)

    fig, axes = plt.subplots(2)
    sample = dataset.__getitem__(0)
    axes[0].imshow(sample['image'])
    axes[1].imshow(sample['semseg'])
    plt.show()
