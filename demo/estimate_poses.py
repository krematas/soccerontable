import argparse
import soccer3d


parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona', help='path')
parser.add_argument('--openpose_dir', default='/home/krematas/code/openpose', help='path')
parser.add_argument('--pad', type=int, default=150, help='pad around players')
parser.add_argument('--height', type=int, default=2160, help='Margin around the pose')
parser.add_argument('--width', type=int, default=3840, help='Margin around the pose')
parser.add_argument('--outscale', type=int, default=4, help='Margin around the pose')
opt, _ = parser.parse_known_args()


db = soccer3d.YoutubeVideo(opt.path_to_data, height=opt.height, width=opt.width)
db.gather_detectron()

db.digest_metadata()

db.get_boxes_from_detectron()
# db.dump_video('detections')

db.estimate_poses(openpose_dir=opt.openpose_dir, pad=opt.pad)
db.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)
db.dump_video('poses', scale=opt.outscale)
