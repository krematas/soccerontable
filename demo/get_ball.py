import argparse
import soccer3d
import numpy as np
from os.path import join

parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Netherlands-Japan-0', help='path')
parser.add_argument('--openpose_dir', default='/home/krematas/code/openpose', help='path')
opt, _ = parser.parse_known_args()


db = soccer3d.YoutubeVideo(opt.path_to_data)
db.gather_detectron()

db.digest_metadata()

db.get_boxes_from_detectron()
db.get_ball_from_detectron(thresh=0.8)


path_to_save = db.path_to_dataset + '/metadata/ball.npy'
np.save(path_to_save, db.ball)

np.save(join(db.path_to_dataset, 'ball', 'poses.npy'), db.poses)

db.dump_video('detections')
