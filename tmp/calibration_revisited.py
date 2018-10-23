import os
import argparse

import soccer3d
import utils.files as file_utils
import utils.io as io


parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/WC10/3', help='path')
opt, _ = parser.parse_known_args()

db = soccer3d.YoutubeVideo(opt.path_to_data, height=2160//2, width=3840//2)
db.gather_detectron()
db.digest_metadata()

i = 0
img = db.get_frame(i)
coarse_mask = db.get_mask_from_detectron(i)
