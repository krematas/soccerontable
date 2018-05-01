import numpy as np
import utils.camera as cam_utils
import utils.files as file_utils
import utils.transform as geo_utils
import utils.draw as draw_utils
from optimization.field_camera_adjustment import calibrate_camera_dist_transf, calibrate_camera_gradient
import matplotlib.pyplot as plt
import cv2
import utils.io as io
import argparse
from os.path import join, exists
from os import listdir
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Track camera given an initial estimate')
parser.add_argument('--dataset', default='kth-0', help='Dataset folder')
parser.add_argument('--frame', type=int, default=0, help='Specific frame to estimate camera parameters')
opt, _ = parser.parse_known_args()

path_to_data = file_utils.get_platform_datadir(join('Singleview/Soccer/', opt.dataset))

path_to_labels = join(path_to_data, 'cnn', 'youtube', 'labels')
path_to_masks = join(path_to_data, 'cnn', 'youtube', 'masks')


label_files = listdir(path_to_masks)
label_files = [i for i in label_files if '_dcrf_2.png' in i]
label_files.sort()

n_files = len(label_files)

for i in range(n_files):
    mask = io.imread(join(path_to_masks, label_files[i]))/255.0
    h, w = mask.shape[:2]

    lbl = np.zeros((2, h, w), dtype=np.float32)
    lbl[0, :, :] = mask[:, :, 0]
    lbl[1, :, :] = mask[:, :, 0]

    savename = label_files[i].replace('_dcrf_2.png', '_r.npy')
    np.save(join(path_to_labels, savename), lbl)
