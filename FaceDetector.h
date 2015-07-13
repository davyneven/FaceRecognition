//
// Created by davy on 3/12/15.
//

#ifndef _RECOGNITIONPROJECT_FACEDETECTOR_H_
#define _RECOGNITIONPROJECT_FACEDETECTOR_H_

#include "opencv2/opencv.hpp"

//include dlib facedetector

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>



using namespace cv;
using namespace std;

#define DESIRED_LEFT_EYE_X 0.16     // Controls how much of the face is visible after preprocessing.
#define DESIRED_LEFT_EYE_Y 0.14
#define FACE_ELLIPSE_CY 0.40
#define FACE_ELLIPSE_W 0.50         // Should be atleast 0.5
#define FACE_ELLIPSE_H 0.80         // Controls how tall the face mask is.

class FaceDetector {

protected:
    dlib::shape_predictor shapePredictor;
    dlib::frontal_face_detector detector;
public:

    FaceDetector(const char *shapePredictorFileName);
        // Search for both eyes within the given face image. Returns the eye centers in 'leftEye' and 'rightEye',
    // or sets them to (-1,-1) if each eye was not found. Note that you can pass a 2nd eyeCascade if you
    // want to search eyes using 2 different cascades. For example, you could use a regular eye detector
    // as well as an eyeglasses detector, or a left eye detector as well as a right eye detector.
    // Or if you don't want a 2nd eye detection, just pass an uninitialized CascadeClassifier.
    // Can also store the searched left & right eye regions if desired.
    void detectBothEyes(const Mat &face, Point &leftEye, Point &rightEye);

    // Histogram Equalizae seperately for the left and right sides of the face,
    // so that if there is a strong light on one side but not the other, it will still look OK.
    void equalizeLeftAndRightHalves(Mat &faceImg);


    // Create a grayscale face image that has a standard size and contrast & brightness.
    // "srcImg" should be a copy of the whole color camera frame, so that it can draw the eye positions onto.
    // If 'doLeftAndRightSeparately' is true, it will process left & right sides seperately,
    // so that if there is a strong light on one side but not the other, it will still look OK.
    // Performs Face Preprocessing as a combination of:
    //  - geometrical scaling, rotation and translation using Eye Detection,
    //  - smoothing away image noise using a Bilateral Filter,
    //  - standardize the brightness on both left and right sides of the face independently using separated Histogram Equalization,
    //  - removal of background and hair using an Elliptical Mask.
    // Returns either a preprocessed face square image or NULL (ie: couldn't detect the face and 2 eyes).
    // If a face is found, it can store the rect coordinates into 'storeFaceRect'

    bool detectFaceAndEyes(Mat &srcImg, Rect *storeFaceRect, Point *storeLeftEye, Point *storeRightEye);

    Mat getPreprocessedFace(Mat &faceImg, int desiredFaceWidth, Point leftEye, Point rightEye);
};


#endif //_RECOGNITIONPROJECT_FACEDETECTOR_H_
