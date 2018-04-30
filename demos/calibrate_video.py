import os

import soccer3d
import utils.files as file_utils


# ----------------------------------------------------------------------------------------------------------------------

db = soccer3d.YoutubeVideo('/home/krematas/data/barcelona')
db.digest_metadata()

file_utils.mkdir(os.path.join(db.path_to_dataset, 'calib'))

db.calibrate_camera()
db.dump_video('calib')
