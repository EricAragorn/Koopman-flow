import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

import pandas as pd
import numpy as np
import os
import pickle
import math
import h5py
from PIL import Image
import io
import re
import cairosvg
import multiprocessing
from tqdm import tqdm
from contextlib import closing

from modules.utils import SubsetDataset

def rgba2rgb(rgba, background=(255,255,255) ):
    c, h, w = rgba.shape

    rgb = torch.tensor(np.zeros((3, h, w)), dtype=torch.float, requires_grad=False)
    r, g, b, a = rgba[0,:,:], rgba[1,:,:], rgba[2,:,:], rgba[3,:,:]

    R, G, B = background

    rgb[0,:,:] = r * a + (1.0 - a) * R
    rgb[1,:,:] = g * a + (1.0 - a) * G
    rgb[2,:,:] = b * a + (1.0 - a) * B

    return rgb

class ADNIMRDataset(Dataset):
    def __init__(self, root, image_index="image_index_hdf5.csv", group=None):
        super(ADNIMRDataset, self).__init__()
        self.root = root
        try:
            self.index_file = pd.read_csv(os.path.join(root, image_index))
        except FileNotFoundError:
            raise ValueError("Image index file {:s} does not exist".format(os.path.join(root, image_index)))

        if group is not None:
            self.index_file = self.index_file[self.index_file['group'].isin(group)].reset_index(drop=True)
        
        f = h5py.File(os.path.join(root, "adni_mri.hdf5"), 'r')
        self.dataset = f['mri']
    
    def _read_img(self, path):
        return np.load(os.path.join(self.root, path))
    
    def __getitem__(self, idx):
        hdf5_idx, subject_id, group, age, sex = self.index_file.loc[idx, ['hdf5_index', 'subject_id', 'group', 'age', 'sex']]

        img = self.dataset[hdf5_idx]

        img = np.expand_dims(img, 0)
        img = np.nan_to_num(img, copy=False, nan=0.)

        return img, (subject_id, group, age, sex)
    
    def __len__(self):
        return len(self.index_file)

class CelebA64Dataset(Dataset):
    def __init__(self, root, transform=None, split='train'):
        super().__init__()
        h5file = os.path.join(root, "celeba/img_align_cropped_64x64_{:s}.hdf5".format(split))
        if not os.path.exists(h5file):
            raise FileNotFoundError("Data file not found under {:s}. \
                                     Please check if you have run the preprocessing script first".format(h5file))
        f = h5py.File(h5file, 'r')
        self.dataset = f['images']
        self.transform = transform
    
    def __getitem__(self, idx):
        img = self.dataset[idx]

        if self.transform is not None:
            img = self.transform(Image.fromarray(img))

        return img, torch.zeros(1,)
    
    def __len__(self):
        return len(self.dataset)

class CelebA128Dataset(Dataset):
    def __init__(self, root, transform=None, split='train'):
        super(CelebA128Dataset, self).__init__()
        h5file = os.path.join(root, "celeba/img_align_cropped_128x128_{:s}.hdf5".format(split))
        if not os.path.exists(h5file):
            raise FileNotFoundError("Data file not found under {:s}. \
                                     Please check if you have run the preprocessing script first".format(h5file))
        f = h5py.File(h5file, 'r')
        self.dataset = f['images']
        self.transform = transform
    
    def __getitem__(self, idx):
        img = self.dataset[idx]

        if self.transform is not None:
            img = self.transform(Image.fromarray(img))

        return img, torch.zeros(1,)
    
    def __len__(self):
        return len(self.dataset)

class ShoeV2Dataset(Dataset):
    def __init__(self, root, split='train', transform_sketch=None, transform_photo=None):
        super().__init__()
        assert split in ['train', 'test']
        self.h5f = h5py.File(os.path.join(root, "ShoeV2/shoev2.hdf5"), 'r')
        self.split = split

        self.transform_sketch = transform_sketch
        self.transform_photo = transform_photo

    def __len__(self):
        return len(self.h5f['{:s}/sketch'.format(self.split)])

    def __getitem__(self, idx):
        sketch = Image.fromarray(self.h5f['{:s}/sketch'.format(self.split)][idx])
        photo = Image.fromarray(self.h5f['{:s}/photo'.format(self.split)][idx])

        if self.transform_sketch is not None:
            sketch = self.transform_sketch(sketch)
        
        sketch = rgba2rgb(sketch) / 255
        
        if self.transform_photo is not None:
            photo = self.transform_photo(photo)
        
        photo = 1.0 - photo # somehow stored as inverted in h5...

        return sketch, photo

def sample_2d_data(dataset, n_samples):

    z = torch.randn(n_samples, 2)

    if dataset == '8gaussians':
        scale = 4
        sq2 = 1/math.sqrt(2)
        centers = [(1,0), (-1,0), (0,1), (0,-1), (sq2,sq2), (-sq2,sq2), (sq2,-sq2), (-sq2,-sq2)]
        centers = torch.tensor([(scale * x, scale * y) for x,y in centers])
        return sq2 * (0.5 * z + centers[torch.randint(len(centers), size=(n_samples,))])

    elif dataset == '2spirals':
        n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * math.pi) / 360
        d1x = - torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
        d1y =   torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
        x = torch.cat([torch.stack([ d1x,  d1y], dim=1),
                       torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3
        return x + 0.1*z

    elif dataset == 'checkerboard':
        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,), dtype=torch.float) * 2
        x2 = x2_ + x1.floor() % 2
        return torch.stack([x1, x2], dim=1) * 2

    elif dataset == 'rings':
        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, set endpoint=False in np; here shifted by one
        linspace4 = torch.linspace(0, 2 * math.pi, n_samples4 + 1)[:-1]
        linspace3 = torch.linspace(0, 2 * math.pi, n_samples3 + 1)[:-1]
        linspace2 = torch.linspace(0, 2 * math.pi, n_samples2 + 1)[:-1]
        linspace1 = torch.linspace(0, 2 * math.pi, n_samples1 + 1)[:-1]

        circ4_x = torch.cos(linspace4)
        circ4_y = torch.sin(linspace4)
        circ3_x = torch.cos(linspace4) * 0.75
        circ3_y = torch.sin(linspace3) * 0.75
        circ2_x = torch.cos(linspace2) * 0.5
        circ2_y = torch.sin(linspace2) * 0.5
        circ1_x = torch.cos(linspace1) * 0.25
        circ1_y = torch.sin(linspace1) * 0.25

        x = torch.stack([torch.cat([circ4_x, circ3_x, circ2_x, circ1_x]),
                         torch.cat([circ4_y, circ3_y, circ2_y, circ1_y])], dim=1) * 3.0

        # random sample
        x = x[torch.randint(0, n_samples, size=(n_samples,))]

        # Add noise
        return x + torch.normal(mean=torch.zeros_like(x), std=0.08*torch.ones_like(x))

    else:
        raise RuntimeError('Invalid `dataset` to sample from.')

def get_dataset(args, transfrom_fn, split='train'):
    if args.dataset == "mnist":
        args.img_dim = (1, 32, 32)
        train_ds = datasets.MNIST("data", train=split == 'train', transform=transfrom_fn(args.img_dim), download=True)
    elif args.dataset == "lsun_bedroom":
        args.img_dim = (3, 256, 256)
        train_ds = SubsetDataset(datasets.LSUN("data/LSUN", classes=['bedroom_train'], transform=transfrom_fn(args.img_dim), target_transform=None), subset_size=60000)
    elif args.dataset == "cifar10":
        args.img_dim = (3, 32, 32)
        train_ds = datasets.CIFAR10("data", train=split == 'train', transform=transfrom_fn(args.img_dim), download=True)
    elif args.dataset == "celeba256":
        args.img_dim = (3, 256, 256)
        train_ds = datasets.CelebA("data", split=split, transform=transfrom_fn(args.img_dim), download=True)
    elif args.dataset == "celeba64":
        args.img_dim = (3, 64, 64)
        train_ds = CelebA64Dataset("data", split=split, transform=transfrom_fn(args.img_dim))
    elif args.dataset == "celeba128":
        args.img_dim = (3, 128, 128)
        train_ds = CelebA128Dataset("data", split=split, transform=transfrom_fn(args.img_dim))
    else:
        raise ValueError("Dataset {:s} not available".format(args.dataset))
    return train_ds

def get_translation_dataset(args, transform_fn_src, transform_fn_tgt, split='train'):
    if args.dataset == "shoev2":
        args.img_dim_src = (3, 64, 64)
        args.img_dim_tgt = (3, 64, 64)
        train_ds = ShoeV2Dataset("data", split=split, transform_sketch=transform_fn_src(args.img_dim_src), transform_photo=transform_fn_tgt(args.img_dim_tgt))
    else:
        raise ValueError("Dataset {:s} not available".format(args.dataset))
    return train_ds

def test():
    dataset = ADNIMRDataset('data/adni_mri', group=['AD'])
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False, pin_memory=True, num_workers=4)
    for i, (X, y) in enumerate(loader):
        print(X.shape)

if __name__ == "__main__":
    test()
