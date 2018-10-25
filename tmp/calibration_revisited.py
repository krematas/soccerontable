import os
import argparse

import soccer3d
import utils.files as file_utils
import utils.io as io
import utils.image as image_utils
import cv2
import utils.camera as cam_utils
import utils.draw as draw_utils
import utils.transform as transf_utils
import numpy as np


parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/WC10/10', help='path')
opt, _ = parser.parse_known_args()

db = soccer3d.YoutubeVideo(opt.path_to_data, height=2160//2, width=3840//2)
db.gather_detectron()
db.digest_metadata()

i = 0
img = db.get_frame(i)
coarse_mask = db.get_mask_from_detectron(i)

edge_sfactor = 0.5
# edge detection
edges = image_utils.robust_edge_detection(cv2.resize(img, None, fx=edge_sfactor, fy=edge_sfactor))
io.imagesc(edges)

cam_data = db.calib[db.frame_basenames[i]]

cam_init = cam_utils.Camera('tmp', cam_data['A'], cam_data['R'], cam_data['T'], db.shape[0], db.shape[1])
template, field_mask = draw_utils.draw_field(cam_init)

II, JJ = (template > 0).nonzero()
synth_field2d = np.array([[JJ, II]]).T[:, :, 0]

field3d = cam_utils.plane_points_to_3d(synth_field2d, cam_init)