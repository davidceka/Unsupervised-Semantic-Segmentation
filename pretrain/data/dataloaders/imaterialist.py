#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import sys
import errno
import hashlib
import glob
import tarfile
import numpy as np
import torch.utils.data as data

from data.util.mypath import Path
from data.util.google_drive import download_file_from_google_drive
from utils.utils import mkdir_if_missing
from PIL import Image


class iMaterialist(data.Dataset):
    def __init__(self, root=Path.db_root_dir('imaterialist-fashion-2020-fgvc7'),
                 saliency='supervised_model',
                 transform=None, overfit=False):
        super(iMaterialist, self).__init__()

        self.root = root
        self.transform = transform

        self.images_dir = os.path.join(self.root, 'images')
        valid_saliency = ['supervised_model', 'unsupervised_model']
        assert(saliency in valid_saliency)
        self.saliency = saliency
        self.sal_dir = os.path.join(self.root, 'saliency_' + self.saliency)

        self.images = []
        self.sal = []

        with open(os.path.join(self.root, '/home/vrai/dataset/imaterialist-fashion-2020-fgvc7/sets/train.txt'), 'r') as f:
            all_ = f.read().splitlines()
        for f in all_:
            _image = os.path.join(self.images_dir, f + ".jpg")
            _sal = os.path.join(self.sal_dir, f + ".png")
            if os.path.isfile(_image) and os.path.isfile(_sal):
                self.images.append(_image)
                self.sal.append(_sal)
        assert (len(self.images) == len(self.sal))    

        if overfit:
            n_of = 32
            self.images = self.images[:n_of]
            self.sal = self.sal[:n_of]
        # Display stats
        print('Number of images: {:d}'.format(len(self.images)))     

    def __getitem__(self, index):
        sample = {}
        sample['image'] = self._load_img(index)
        sample['sal'] = self._load_sal(index)

        if self.transform is not None:
            sample = self.transform(sample)
        
        sample['meta'] = {'image': str(self.images[index])}

        return sample

    def __len__(self):
            return len(self.images)

    def _load_img(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        w, h = _img.size
        basesize = 250
        if h > basesize or w > basesize:
            if(w>h):
                wpercent = (basesize/float(w))
                hsize = int((float(h)*float(wpercent)))
                _img = _img.resize((basesize,hsize), Image.ANTIALIAS)
            else:
                hpercent=(basesize/float(h))
                wsize=int((float(w)*float(hpercent)))
                _img = _img.resize((wsize,basesize), Image.ANTIALIAS)
        
        # _img = _img.resize((int(w / 7.26), int(h / 7.26)))
        return _img

    def _load_sal(self, index):
        _sal = Image.open(self.sal[index])
        w, h = _sal.size

        basesize = 250
        if h > basesize or w > basesize:
            if(w>h):
                wpercent = (basesize/float(w))
                hsize = int((float(h)*float(wpercent)))
                _sal = _sal.resize((basesize,hsize), Image.ANTIALIAS)
            else:
                hpercent=(basesize/float(h))
                wsize=int((float(w)*float(hpercent)))
                _sal = _sal.resize((wsize,basesize), Image.ANTIALIAS)

        #_sal = _sal.resize((int(w/7.26), int(h/7.26)))
        return _sal

    def __str__(self):
        return 'imaterialist-fashion-2020-fgvc7(saliency=' + self.saliency + ')'

    def get_class_names(self):
        # Class names for sal
        return ['background', 'salient object']

if __name__ == '__main__':
    """ For purpose of debugging """
    from matplotlib import pyplot as plt
    # Sample from supervised saliency model
    dataset = iMaterialist(saliency='supervised_model')
    sample = dataset.__getitem__(5)
    fig, axes = plt.subplots(2)
    axes[0].imshow(sample['image'])
    axes[1].imshow(sample['sal'])
    plt.show()
    plt.close()

    # Sample from unsupervised saliency model
    dataset = iMaterialist(saliency='unsupervised_model')
    sample = dataset.__getitem__(5)
    fig, axes = plt.subplots(2)
    axes[0].imshow(sample['image'])
    axes[1].imshow(sample['sal'])
    plt.show()
    plt.close()
