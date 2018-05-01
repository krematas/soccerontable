import torch
import numpy as np
from scipy.misc import imread, imresize, imsave, imrotate
from torchvision import transforms
import scipy.ndimage
import utils.io as io

__all__ = ['NormalizeImage', 'ColorOffset', 'Rescale', 'RandomCrop', 'RandomRotation', 'ToTensor', 'LabelToVolume']


# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
normalize = transforms.Normalize(mean=[0.3402085, 0.42575407, 0.23771574], std=[0.1159472, 0.10461029, 0.13433486])


class NormalizeImage(object):
    def __call__(self, sample):
        image, mask, target = sample['image'], sample['mask'], sample['target']
        image = normalize(image)
        return {'image': image, 'mask': mask, 'target': target}


class ColorOffset(object):
    def __call__(self, sample):
        image, mask, target = sample['image'], sample['mask'], sample['target']

        r_offset, g_offset, b_offset = np.random.randint(-10, 10, 3).astype(np.uint8)

        image[:, :, 0] += r_offset
        image[:, :, 1] += g_offset
        image[:, :, 2] += b_offset

        image = np.maximum(image, 0)
        image = np.minimum(image, 255)

        return {'image': image, 'mask': mask, 'target': target}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, img_size, label_size):
        assert isinstance(img_size, int)
        self.img_size = img_size
        self.label_size = label_size

    def __call__(self, sample):
        image, mask, target = sample['image'], sample['mask'], sample['target']
        image = imresize(image, (self.img_size, self.img_size))
        mask = imresize(mask, (self.img_size, self.img_size), interp='nearest', mode='F')
        target = imresize(target, (self.label_size, self.label_size), interp='nearest', mode='F')

        return {'image': image, 'mask': mask, 'target': target}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __call__(self, sample):
        image, mask, target = sample['image'], sample['mask'], sample['target']

        h, w = image.shape[:2]

        x1, y1 = np.random.uniform(0, 0.2, 2)
        x2, y2 = np.random.uniform(0.8, 1, 2)
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        image = image[y1:y2, x1:x2, :]
        mask = mask[y1:y2, x1:x2]
        target = target[y1:y2, x1:x2]

        return {'image': image, 'mask': mask, 'target': target}


class RandomRotation(object):

    def __call__(self, sample, interval=(-10, 10)):
        image, mask, target = sample['image'], sample['mask'], sample['target']

        angle = np.random.uniform(interval[0], interval[1])

        image = imrotate(image, angle)
        mask = imrotate(mask, angle, interp='nearest', mode='F')
        target = imrotate(target, angle, interp='nearest', mode='F')

        return {'image': image, 'mask': mask, 'target': target}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, transpose_target=True):
        self.transpose_target = transpose_target

    def __call__(self, sample):
        image, mask, target = sample['image'], sample['mask'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))/255.0
        mask = mask[:, :, None].transpose((2, 0, 1))
        if self.transpose_target:
            target = target[:, :, None].transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image)
        image_tensor = torch.FloatTensor(image_tensor.size()).copy_(image_tensor)
        return {'image': image_tensor, 'mask': torch.from_numpy(mask), 'target': torch.from_numpy(target)}


class LabelToVolume(object):

    def __init__(self, nbins, sigma=1.5):
        self.nbins = nbins
        self.sigma = sigma

    def __call__(self, sample):
        image, mask, target = sample['image'], sample['mask'], sample['target']

        h, w = target.shape[:2]
        I, J = (target != 0).nonzero()
        C = target[I, J].ravel().astype(int)

        label_cube = np.zeros((h, w, self.nbins), dtype=np.float32)
        label_cube[I, J, C] = 1.0

        label_cube_smoothed = np.zeros_like(label_cube)

        denom = (1. / (np.sqrt(2 * np.pi * self.sigma ** 2)))
        for i, j in zip(I, J):
            input_signal = label_cube[i, j, 1:]
            output_signal = scipy.ndimage.filters.gaussian_filter1d(input_signal, self.sigma) / denom
            label_cube_smoothed[i, j, 1:] = output_signal

        I, J = (target == 0).nonzero()
        label_cube_smoothed[I, J, 0] = 1
        label_cube_smoothed = np.transpose(label_cube_smoothed, (2, 0, 1))
        return {'image': image, 'mask': mask, 'target': label_cube_smoothed}

# def preprocess_img(img):
#     # [0,255] image to [0,1]
#     min = img.min()
#     max = img.max()
#     img = torch.FloatTensor(img.size()).copy_(img)
#     img.add_(-min).mul_(1.0 / (max - min))
#
#     # RGB to BGR
#     idx = torch.LongTensor([2, 1, 0])
#     img = torch.index_select(img, 0, idx)
#
#     # [0,1] to [-1,1]
#     img = img.mul_(2).add_(-1)
#
#     # check that input is in expected range
#     assert img.max() <= 1, 'badly scaled inputs'
#     assert img.min() >= -1, "badly scaled inputs"
#
#     return img
