//
// Created by krematas on 3/9/18.
//

//
// Created by krematas on 3/9/18.
//
#include <Eigen/Sparse>
#include "core.h"

typedef Eigen::SparseMatrix<var_t> SpMat;
typedef Eigen::Triplet<var_t> T;

void getPixelNeighbors(int height, int width, std::vector<std::vector<int>>& neighborId){

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){

            if(i == 0){
                neighborId[i*width+j].push_back((i+1)*width+j);
            }else if(i == height-1){
                neighborId[i*width+j].push_back((i-1)*width+j);
            }else{
                neighborId[i*width+j].push_back((i+1)*width+j);
                neighborId[i*width+j].push_back((i-1)*width+j);
            }

            if(j == 0){
                neighborId[i*width+j].push_back(i*width+j+1);
            }else if(j == width-1){
                neighborId[i*width+j].push_back(i*width+j-1);
            }else{
                neighborId[i*width+j].push_back(i*width+j+1);
                neighborId[i*width+j].push_back(i*width+j-1);
            }

        }
    }

}


void getLabelPosition(var_t *img, int h, int w, std::map<int, std::vector<int>>& ht){
    for(int i=0; i<h; i++) {
        for (int j = 0; j < w; j++) {

            if(img[i*w+j] >= 1.0){
                int lbl = int(img[i*w+j]-1);
                ht[lbl].push_back(i*w+j);
            }

        }
    }
}


SpMat setU(int N, std::map<int, std::vector<int>>& ht, Eigen::VectorXf& y){
    std::vector<T> tripletList;

    for(std::map<int,std::vector<int>>::iterator it = ht.begin(); it != ht.end(); ++it) {
        std::vector<int> pixLocation = it->second;
        for(int i=0; i<pixLocation.size();i++){
            tripletList.push_back(T(pixLocation[i], pixLocation[i], 1.));
            y[pixLocation[i]] = float(it->first);
        }
    }

    SpMat U(N,N);
    U.setFromTriplets(tripletList.begin(), tripletList.end());
    return U;
}

void setDW(var_t* image, var_t* edges, int h, int w, std::vector<SpMat>& out, float sigma1, float sigma2){

    int N = h * w;
    std::vector<std::vector<int>> neighborId(N);
    getPixelNeighbors(h, w, neighborId);

    std::vector<T> tripletListD;
    std::vector<T> tripletListW;
    int M = 0;

    for(int i=0; i<neighborId.size(); i++){
        int x, y;
        y = i/w;
        x = i%w;
        var_t r1 = image[y*w*3+x*3+0];
        var_t g1 = image[y*w*3+x*3+1];
        var_t b1 = image[y*w*3+x*3+2];
        var_t e1 = edges[y*w+x];

        for(int j=0; j<neighborId[i].size(); j++){

            y = neighborId[i][j]/w;
            x = neighborId[i][j]%w;
            var_t r2 = image[y*w*3+x*3+0];
            var_t g2 = image[y*w*3+x*3+1];
            var_t b2 = image[y*w*3+x*3+2];

            var_t weight0 = exp(-((r1-r2)*(r1-r2) + (g1-g2)*(g1-g2)+ (b1-b2)*(b1-b2))/sigma1);
            var_t weight1 = exp(-(e1*e1)/sigma2);

            tripletListD.push_back(T(M, i, 1.));
            tripletListD.push_back(T(M, neighborId[i][j], -1.));
            tripletListW.push_back(T(M, M, weight0*weight1));

            M++;

        }
    }

    SpMat D(M, N);
    D.setFromTriplets(tripletListD.begin(), tripletListD.end());

    SpMat W(M, M);
    W.setFromTriplets(tripletListW.begin(), tripletListW.end());
    out.push_back(D);
    out.push_back(W);
}


var_t* segmentFromPoses(var_t *img, var_t *edges, var_t *poseData, int height, int width, float sigma1, float sigma2){
    std::map<int, std::vector<int>> ht;
    getLabelPosition(poseData, height, width, ht);
    Eigen::VectorXf y(height*width);
    SpMat U = setU(height*width, ht, y);
    std::vector<SpMat> DW;
    setDW(img, edges, height, width, DW, sigma1, sigma2);
    SpMat D = DW[0];
    SpMat W = DW[1];

    Eigen::VectorXf b = U*y;

    SpMat A = U + D.transpose()*W*D;

    Eigen::SimplicialCholesky <SpMat> solver(A);
    Eigen::VectorXf x = solver.solve(b);

    var_t *output = new var_t[height*width];
    for(int i=0; i<height*width; i++)
        output[i] = x[i];

    return output;
}