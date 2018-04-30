import os

import soccer3d
import utils.files as file_utils


# ----------------------------------------------------------------------------------------------------------------------
dataset = 'b5'

db = soccer3d.YoutubeVideo(file_utils.get_platform_datadir('Multiview/Portland/{0}'.format(dataset)))
db.digest_metadata()

# file_utils.mkdir(os.path.join(db.path_to_dataset, 'calib'))
#
# db.calibrate_camera()
# db.dump_video('calib')
