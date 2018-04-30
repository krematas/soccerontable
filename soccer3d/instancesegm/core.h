//
// Created by krematas on 3/9/18.
//

#ifndef LAPLACIAN_CORE_H
#define LAPLACIAN_CORE_H

#endif //LAPLACIAN_CORE_H

//
// Created by krematas on 3/9/18.
//

#include <vector>
#include <stdio.h>
#include <math.h>
#include <iostream>

typedef float var_t;

var_t* segmentFromPoses(var_t *img, var_t *edges, var_t *poseData, int height, int width, float sigma1, float sigma2);
