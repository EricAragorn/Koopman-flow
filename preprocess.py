import h5py
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import cairosvg
import re
from PIL import Image
import io
import multiprocessing

def process_celeba_data(data_root, h5file, split='training', side_length=None):

    dir_path = os.path.join(data_root, 'celeba/img_align_celeba')
    assert os.path.exists(dir_path)
    filelist = [filename for filename in os.listdir(dir_path) if filename.endswith('jpg')]
    assert len(filelist) == 202599
    if split == 'training':
        start_idx, end_idx = 0, 162770
    elif split == 'val':
        start_idx, end_idx = 162770, 182637
    else:
        start_idx, end_idx = 182637, 202599

    if type(h5file) == str:
        f = h5py.File(os.path.join(data_root, "celeba", h5file), 'w')
    elif type(h5file) == h5py.File:
        f = h5file
    else:
        raise ValueError("h5file argument should be either a string or a h5py File object\n")

    if side_length is None:
        dset = f.create_dataset("images", (end_idx - start_idx, 128, 128, 3), dtype=np.uint8)
    else:
        dset = f.create_dataset("images", (end_idx - start_idx, side_length, side_length, 3), dtype=np.uint8)

    for i in tqdm(range(end_idx - start_idx)):
        img = np.array(imread(os.path.join(dir_path, filelist[i + start_idx])))
        img = img[45:173,25:153]
        if side_length is not None:
            img = resize(img, [side_length, side_length])
        dset[i] = (img * 255).astype(np.uint8)

class _GeneralImageMapper:
    def __init__(self, dataset, path, target_size=None):
        self.path = path
        self.dataset = dataset
        self.target_size = target_size

    def __call__(self, idx):
        img = imread(self.path[idx]) * 255
        if self.target_size is not None:
            img = resize(img.astype(np.float), self.target_size).round().astype(np.uint8)
        self.dataset[idx] = img

class _SVGMapper(_GeneralImageMapper):

    def __call__(self, idx):
        data_io = io.BytesIO()
        cairosvg.svg2png(url=self.path[idx], write_to=data_io)
        img = np.asarray(Image.open(data_io), dtype=np.uint8)
        data_io.close()
        if self.target_size is not None:
            img = resize(img.astype(np.float), self.target_size).round().astype(np.uint8)
        self.dataset[idx] = img
    

if __name__ == "__main__":
    process_celeba_data("./data", "img_align_cropped_64x64_train.hdf5", split='train', side_length=64)
    process_celeba_data("./data", "img_align_cropped_64x64_test.hdf5", split='test', side_length=64)