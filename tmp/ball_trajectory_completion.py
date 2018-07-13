import argparse
import soccer3d
import numpy as np
import cv2
from matplotlib import pyplot as plt
import utils.io as io

parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Japan-Something-0', help='path')
parser.add_argument('--openpose_dir', default='/home/krematas/code/openpose', help='path')
opt, _ = parser.parse_known_args()


db = soccer3d.YoutubeVideo(opt.path_to_data)
db.gather_detectron()

db.digest_metadata()

db.get_boxes_from_detectron()
db.get_ball_from_detectron(thresh=0.80)

# extract balls
for fname in db.frame_basenames:
    if len(db.ball[fname]) > 0:
        # img = db.get_frame(db.frame_basenames.index(fname))

        scores = db.ball[fname][:, 4]
        db.ball[fname] = np.array([db.ball[fname][np.argmax(scores), :]])
        bbox = db.ball[fname]
        # if len(scores)> 1:
        #     io.show_box(img, bbox)
        # print(scores)
        x1, y1, x2, y2 = bbox[0, :4].astype(int)
        # ball_img.append(img[y1:y2, x1:x2, :])
        # ball_coord.append([x1, y1, x2, y2])


fname = db.frame_basenames[0]
img = db.get_frame(0)
bbox = db.ball[fname]
x1, y1, x2, y2 = bbox[0, :4].astype(int)
ball_img = img[y1:y2, x1:x2, :]
cur_pos = [x1, y1]

margin = 50

for i, fname in enumerate(db.frame_basenames):
    if len(db.ball[fname]) == 0:
        img = db.get_frame(db.frame_basenames.index(fname))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_ball = cv2.cvtColor(ball_img, cv2.COLOR_RGB2GRAY)

        x1, y1 = cur_pos
        x1 -= margin
        y1 -= margin
        x2, y2 = x1 + 2*margin, y1 + 2*margin

        res = cv2.matchTemplate(gray[y1:y2, x1:x2], gray_ball, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + ball_img.shape[1], top_left[1] + ball_img.shape[0])

        bbox = np.array([[top_left[0]+x1, top_left[1]+y1, bottom_right[0]+x1, bottom_right[1]+y1, 1]])

        x1, y1, x2, y2 = bbox[0, :4].astype(int)
        ball_img = img[y1:y2, x1:x2, :]
        cur_pos = [x1, y1]

        # io.imagesc(res)
        db.ball[fname] = bbox
        # io.show_box(img, bbox)
        # io.show_box(img, np.array([[x1, y1, x2, y2, 1]]))
        # break

    else:

        fname = db.frame_basenames[i]
        img = db.get_frame(i)
        bbox = db.ball[fname]
        x1, y1, x2, y2 = bbox[0, :4].astype(int)
        ball_img = img[y1:y2, x1:x2, :]
        cur_pos = [x1, y1]

db.dump_video('detections', scale=2)

for fname in db.frame_basenames:
    if len(db.ball[fname]) == 0:
        print(fname)

# img = cv.imread('messi5.jpg',0)
# img2 = img.copy()
# template = cv.imread('template.jpg',0)
# w, h = template.shape[::-1]
# # All the 6 methods for comparison in a list
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
#             'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
# for meth in methods:
#     img = img2.copy()
#     method = eval(meth)
#     # Apply template Matching
#     res = cv.matchTemplate(img,template,method)
#     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#     cv.rectangle(img,top_left, bottom_right, 255, 2)
#     plt.subplot(121),plt.imshow(res,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()