# <b>Soccer On Your Tabletop</b> [[Project Page]](grail.cs.washington.edu/projects/soccer/)

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
Install [Detectron](https://github.com/facebookresearch/Detectron) for detection and (instance) segmentation.

Install [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for the players' pose estimation.


-----------------

## Upconversion of YouTube Soccer videos to 3D ##
