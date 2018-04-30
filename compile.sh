#!/usr/bin/env bash
cur=$PWD
cd utils/nms
make -j4
cd ${cur}

cd soccer3d/instancesegm
mkdir build
cd build
cmake ..
make -j4
mv instancesegm ../
cd ..
rm -rf build
cd ${cur}
