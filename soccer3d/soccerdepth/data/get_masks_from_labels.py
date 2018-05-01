import numpy as np
import os
import utils.files as file_utils
import cv2
import utils.io as io
import argparse
from os.path import join, exists
from os import listdir
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Track camera given an initial estimate')
parser.add_argument('--dataset', default='Barcelona-Atlentico-0', help='Dataset folder')
opt, _ = parser.parse_known_args()

path_to_data = file_utils.get_platform_datadir(join('Singleview/Soccer/', opt.dataset))

path_to_labels = join(path_to_data, 'cnn', 'youtube', 'labels')
path_to_masks = join(path_to_data, 'cnn', 'youtube', 'masks')
if not exists(path_to_masks):
    os.mkdir(path_to_masks)


label_files = listdir(path_to_labels)
label_files = [i for i in label_files if '_r.npy' in i]
label_files.sort()

n_files = len(label_files)

for i in range(n_files):
    lbl = np.load(join(path_to_labels, label_files[i]))
    savename = label_files[i].replace('_r.npy', '_dcrf_2.png')
    cv2.imwrite(join(path_to_masks, savename), lbl[:, :, 0]*255)
