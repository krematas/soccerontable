import json
import matplotlib.pyplot as plt
import cv2
import numpy as np

datadir = '/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Russia2018/adnan-januzaj-goal-england-v-belgium-match-45/tmp/'
json_file = datadir + '0_keypoints.json'
with open(json_file) as data_file:
    # for iii in range(2):
    #     _ = data_file.readline()
    data_json = json.load(data_file)

    # if len(data_json['people']) == 0:
    #     continue
    # sz = data_json['sizes']
    n_persons = len(data_json['people'])
    # keypoints = np.array(data_json['data']).reshape(sz)

    for k in range(n_persons):
        keypoints_ = np.array(data_json['people'][k]['pose_keypoints_2d']).reshape((25, 3))

img = cv2.imread(datadir + '0.jpg')[:, :, ::-1]
plt.imshow(img)
plt.plot(keypoints_[:, 0], keypoints_[:, 1], 'ro')
plt.show()