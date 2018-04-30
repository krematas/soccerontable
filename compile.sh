cur=$PWD
cd utils/nms
make
cd ${cur}

cd utils/cocoapi/PythonAPI
make
cd ${cur}

