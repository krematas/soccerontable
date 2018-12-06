import os
import argparse

import soccer3d
import utils.files as file_utils


parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018/denis-cheryshev-goal-russia-egypt', help='path')
parser.add_argument('--height', type=int, default=2160, help='Margin around the pose')
parser.add_argument('--width', type=int, default=3840, help='Margin around the pose')
parser.add_argument('--fps', type=int, default=50, help='Margin around the pose')


opt, _ = parser.parse_known_args()

db = soccer3d.YoutubeVideo(opt.path_to_data, height=opt.height, width=opt.width)
db.gather_detectron()
db.digest_metadata()

file_utils.mkdir(os.path.join(db.path_to_dataset, 'calib'))

db.calibrate_camera()
db.dump_video('calib', fps=opt.fps)
