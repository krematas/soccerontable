from os import listdir
from os.path import join
import numpy as np
import torch.utils.data as data
from scipy.misc import imread


def get_set(train_dir, nbins=-1, transform=None, additional_input_type='estmask'):
    return DatasetFromFolder(train_dir, bins=nbins, transform=transform, additional_input_type=additional_input_type)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy", ".npz"])

ADDITIONAL_INPUT = {'trimap': '_dcrf_tri.png', 'estmask': '.png'}


class DatasetFromFolder(data.Dataset):

    def __init__(self, data_dir, bins=-1, transform=None, additional_input_type='estmap'):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(data_dir, 'images',  x) for x in listdir(join(data_dir, 'images')) if is_image_file(x)]
        self.image_filenames.sort()
        self.target_filenames = [x.replace('images', 'labels').replace('.jpg', '.npy') for x in self.image_filenames]
        self.mask_filenames = [x.replace('images', 'masks').replace('.jpg', ADDITIONAL_INPUT[additional_input_type]) for x in self.image_filenames]

        self.bins = bins
        self.transform = transform

    def __getitem__(self, index):

        input = load_img(self.image_filenames[index])
        depth, mask_gt = load_label(self.target_filenames[index], nbins=self.bins)
        mask = load_custom_mask(self.mask_filenames[index])
        sample = {'image': input, 'mask': mask, 'target': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_filenames)


# ----------------------------------------------------------------------------------------------------------------------

def load_custom_mask(filepath):
    mask = imread(filepath)/255.
    return mask


def load_img(filepath):
    img = imread(filepath)
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
    return img


def load_label(filepath, nbins=-1):
    label = np.load(filepath).item()
    depth, mask, billboard = label['depth'], label['mask'], label['billboard']
    depth -= billboard
    if nbins > 0:
        bins = np.linspace(-0.5, 0.5, nbins-2)
        depth = (np.digitize(depth, bins)+1)*mask

    return depth, mask
