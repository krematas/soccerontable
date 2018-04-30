# <b>Soccer On Your Tabletop</b> [[Project Page]](http://grail.cs.washington.edu/projects/soccer/)

[Konstantinos Rematas](https://homes.cs.washington.edu/~krematas/), [Ira Kemelmacher-Shlizerman](https://homes.cs.washington.edu/~kemelmi/), [Brian Curless](https://homes.cs.washington.edu/~curless/), [Steve Seitz](https://homes.cs.washington.edu/~seitz/), in CVPR 2018

![Teaser Image](http://grail.cs.washington.edu/projects/soccer/images/teaser.png)


Warning: I am in the process of transferring the repo, so many things will probably not work.

-----------------

### Overview ###
This repository contains:

Upconversion of YouTube Soccer videos to 3D
  - Camera Calibration
  - Player Analysis (detection/segmentation/tracking)
  - Player Depth estimation
  - Temporal Game Reconstruction

Scripts for getting training data from video games
  - Electronic Arts FIFA 2016 RenderDoc Depth buffer capture
  - Depth buffer to point cloud

Visualization tools
  - Example Unity project
  - Hololens VS Solution

### Dependencies ###
These are the non "pip3 install" dependencies:
- [Detectron](https://github.com/facebookresearch/Detectron) for detection and (instance) segmentation.
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for the players' pose estimation.
- [OpenCV 3.1](https://github.com/opencv/opencv) + [OpenCV_contrib](https://github.com/opencv/opencv_contrib) for image loading/edge estimation etc. I followed [this](https://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/) guide and it worked fine.
- [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) for the instance segmentation
- [Cocoapi](https://github.com/cocodataset/cocoapi) for mask utilities
- Boost
- CMake


-----------------

## Upconversion of YouTube Soccer videos to 3D ##
The pipeline contains parts written or depending on python2, python3, cython, 
C/C++, which makes it a bit difficult to combine everything in one system. 
Therefore we break it into individual parts that have specific inputs and outputs 
(eg png to pickle) and communicate through a python3 class that reads, processes 
and writes the intermediate and final results.  

Let's start by downloading an example dataset
```
wget http://grail.cs.washington.edu/projects/soccer/barcelona.zip
unzip barcelona.zip
# DATADIR=/path/to/barcelona
```

The original video was cropped from [YouTube](https://www.youtube.com/watch?v=hYU51XQruq0) 
and frames were extracted with avconv.

Run Detectron to get bounding boxes and segmentation masks
```
mkdir $DATADIR/detectron
# DETECTRON=/path/to/clone/detectron
cp utils/thirdpartyscripts/infer_subimages.py ./$DETECTRON/tools/
cd $DETECTRON
python2 tools/infer_subimages.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml --output-dir $DATADIR/detectron --image-ext jpg --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl $DATADIR/images/
```