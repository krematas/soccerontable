import os
import argparse

import soccer3d
import utils.files as file_utils


parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona', help='path')
opt, _ = parser.parse_known_args()

db = soccer3d.YoutubeVideo(opt.path_to_data)
db.gather_detectron()
db.digest_metadata()

file_utils.mkdir(os.path.join(db.path_to_dataset, 'calib'))

db.calibrate_camera()
db.dump_video('calib')
