#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <ctime>
#include "core.h"
#include "boost/filesystem.hpp"   // includes all needed Boost.Filesystem declarations
#include <boost/program_options.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <iostream>               // for std::cout
#include <Eigen/Core>

using namespace boost::filesystem;
using namespace boost::program_options;

using namespace cv;
using namespace cv::ximgproc;

int main(int argc, char** argv )
{
    Eigen::initParallel();
    int nthreads = Eigen::nbThreads( );

    options_description desc{"Options"};
    desc.add_options()
            ("help,h", "Help screen")
            ("path_to_data", value<std::string>()->default_value("./"), "Path to dataset")
            ("path_to_model", value<std::string>()->default_value("./model.yml.gz"), "Path to model")
            ("extension", value<std::string>()->default_value(".jpg"), "File extension")
            ("s1", value<float>()->default_value(1.0), "Sigma1 (RGB)")
            ("s2", value<float>()->default_value(0.01), "Sigma2 (edges)")
            ("thresh", value<float>()->default_value(1.5), "Threshold");

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if (vm.count("help"))
        std::cout << desc << '\n';

    path p (vm["path_to_data"].as<std::string>()+"/images/");

    directory_iterator end_itr;

    std::vector<std::string> fileList;
    for (directory_iterator itr(p); itr != end_itr; ++itr)
    {
        if (is_regular_file(itr->path())) {
            std::string current_file = itr->path().string();
            if(extension(current_file)==vm["extension"].as<std::string>()){
                fileList.push_back(current_file);
            }

        }
    }

    float sigma1 = vm["s1"].as<float>();
    float sigma2 = vm["s2"].as<float>();
    float thresh = vm["thresh"].as<float>();

    cv::Ptr<StructuredEdgeDetection> pDollar = createStructuredEdgeDetection(vm["path_to_model"].as<std::string>());

    #pragma omp parallel for
    for(int f=0; f<fileList.size(); f++){
        clock_t begin = clock();

        std::string current_imagename = fileList[f];
        std::string current_posename = fileList[f];
        boost::replace_all(current_posename, vm["extension"].as<std::string>(), ".png");
        boost::replace_all(current_posename, "images", "poseimgs");

        Mat image, poseImage;
        image = imread( current_imagename, 1 );
        image.convertTo(image, cv::DataType<float>::type, 1/255.0);
        var_t *imgData = (var_t*)(image.data);

        poseImage = imread( current_posename, 0 );
        poseImage.convertTo(poseImage, cv::DataType<float>::type);
        var_t *poseData = (var_t*)(poseImage.data);

        Mat img2;
        image.copyTo(img2);
        img2.convertTo(img2, cv::DataType<var_t>::type);
        cv::Mat edges(img2.size(), img2.type());

        pDollar->detectEdges(img2, edges);

        int height = image.rows;
        int width = image.cols;

        var_t *edgesData = (var_t*)(edges.data);
        var_t* output = segmentFromPoses(imgData, edgesData, poseData, height, width, sigma1, sigma2);


        Mat outimg(height, width, CV_8U);
        for(int i=0; i<height; i++) {
            for (int j = 0; j < width; j++) {
                if(output[i*width+j] > thresh)
                    outimg.at<uchar>(i,j) = 255;
                else
                    outimg.at<uchar>(i,j) = 0;

            }
        }

        std::string outname = fileList[f];
        boost::replace_all(outname, vm["extension"].as<std::string>(), ".png");
        boost::replace_all(outname, "images", "pose_masks");
        imwrite(outname, outimg);

//        clock_t end = clock();
//        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//        std::cout<<"Total time: "<<elapsed_secs<<std::endl;
//        namedWindow("Display Image", WINDOW_AUTOSIZE );
//        imshow("Display Image", outimg);
//        waitKey(0);

    }






    return 0;
}
