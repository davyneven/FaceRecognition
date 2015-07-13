//
// Created by davy on 3/14/15.
//

#ifndef _RECOGNITIONPROJECT_RECOGNITION_H_
#define _RECOGNITIONPROJECT_RECOGNITION_H_

#include <stdio.h>
#include <iostream>
#include <vector>

// Include OpenCV's C++ Interface
#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;


class Recognition {

protected:
        Ptr<FaceRecognizer> model;
public:
        Recognition(const char *facerecAlgorithm = "FaceRecognizer.Eigenfaces");

    // Start training from the collected faces.
    // The face recognition algorithm can be one of these and perhaps more, depending on your version of OpenCV, which must be atleast v2.4.1:
    //    "FaceRecognizer.Eigenfaces":  Eigenfaces, also referred to as PCA (Turk and Pentland, 1991).
    //    "FaceRecognizer.Fisherfaces": Fisherfaces, also referred to as LDA (Belhumeur et al, 1997).
    //    "FaceRecognizer.LBPH":        Local Binary Pattern Histograms (Ahonen et al, 2006).
        void learnCollectedFaces(const vector<Mat> preprocessedFaces, const vector<int> faceLabels);

    // Show the internal face recognition data, to help debugging.
        void showTrainingDebugData(const int faceWidth, const int faceHeight);

    // Generate an approximately reconstructed face by back-projecting the eigenvectors & eigenvalues of the given (preprocessed) face.
        Mat reconstructFace(const Mat preprocessedFace);

    // Compare two images by getting the L2 error (square-root of sum of squared error).
        double getSimilarity(const Mat A, const Mat B);

        int predict(const Mat preprocessedFace);

        Mat getImageFrom1DFloatMat(const Mat matrixRow, int height);

        bool loadModel(const char* fileName);

        void safeModel(const char *fileName);

};


#endif //_RECOGNITIONPROJECT_RECOGNITION_H_
