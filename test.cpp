//
// Created by davy on 3/12/15.
//


const char *shapePredictorFileName = "/home/davy/shape_predictor_68_face_landmarks.dat";
const char *facerecAlgorithm = "FaceRecognizer.Eigenfaces";


// Set the desired face dimensions. Note that "getPreprocessedFace()" will return a square face.
const int faceWidth = 70;
const int faceHeight = faceWidth;

// Try to set the camera resolution. Note that this only works for some cameras on
// some computers and only for some drivers, so don't rely on it to work!
const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;


const char *windowName = "WebcamFaceRec";   // Name shown in the GUI window.
const int BORDER = 8;  // Border between GUI elements to the edge of the image.

#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>

// Include OpenCV's C++ Interface
#include <opencv2/opencv.hpp>

//include dlib
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>

// Include the rest of our code!
#include "recognition.h"     // Train the face recognition system and recognize a person from an image.
#include "FaceDetector.h"


using namespace cv;
using namespace std;


#if !defined VK_ESCAPE
#define VK_ESCAPE 0x1B      // Escape character (27)
#endif


// Running mode for the Webcam-based interactive GUI program.
enum MODES {MODE_STARTUP=0, MODE_DETECTION, MODE_COLLECT_FACES, MODE_TRAINING, MODE_RECOGNITION, MODE_DELETE_ALL, MODE_END};
const char* MODE_NAMES[] = {"Startup", "Detection", "Collect Faces", "Training", "Recognition", "Delete All", "ERROR!"};
MODES m_mode = MODE_STARTUP;

FaceDetector *faceDetector;


// Get access to the webcam.#include <opencv2/contrib/contrib.hpp>
void initWebcam(VideoCapture &videoCapture, int cameraNumber)
{
    // Get access to the default camera.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        videoCapture.open(cameraNumber);
    } catch (cv::Exception &e) {}
    if ( !videoCapture.isOpened() ) {
        cerr << "ERROR: Could not access the camera!" << endl;
        exit(1);
    }
    cout << "Loaded camera " << cameraNumber << "." << endl;
}




// Main loop that runs forever, until the user hits Escape to quit.
void recognizeAndTrainUsingWebcam(VideoCapture &videoCapture)
{
    Ptr<FaceRecognizer> model;

    // Since we have already initialized everything, lets start in Detection mode.
    m_mode = MODE_DETECTION;


    // Run forever, until the user hits Escape to "break" out of this loop.
    while (true) {

        // Grab the next camera frame. Note that you can't modify camera frames.
        Mat cameraFrame;
        videoCapture >> cameraFrame;
        if( cameraFrame.empty() ) {
            cerr << "ERROR: Couldn't grab the next camera frame." << endl;
            exit(1);
        }

        // Get a copy of the camera frame that we can draw onto.
        Mat displayedFrame;
        cameraFrame.copyTo(displayedFrame);

        // Find a face and preprocess it to have a standard size and contrast & brightness.
        Rect faceRect;  // Position of detected face.
        Point leftEye, rightEye;    // Position of the detected eyes.
        bool gotFaceAndEyes = faceDetector->detectFaceAndEyes(displayedFrame,&faceRect, &leftEye, &rightEye);
        //Mat preprocessedFace = faceDetector-> getPreprocessedFace(displayedFrame, faceWidth, &faceRect);

        // Draw an anti-aliased rectangle around the detected face.
        if (gotFaceAndEyes) {
            rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);

            // Draw light-blue anti-aliased circles for the 2 eyes.
            Scalar eyeColor = CV_RGB(0,255,255);
            if (leftEye.x >= 0) {   // Check if the eye was detected
                circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
            }
            if (rightEye.x >= 0) {   // Check if the eye was detected
                circle(displayedFrame, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);
            }
        }

        if (m_mode == MODE_DETECTION) {
            // Don't do anything special.
        }

        if (m_mode = MODE_TRAINING) {

        }
        // Show the camera frame on the screen.
        imshow(windowName, displayedFrame);


        // IMPORTANT: Wait for atleast 20 milliseconds, so that the image can be displayed on the screen!
        // Also checks if a key was pressed in the GUI window. Note that it should be a "char" to support Linux.
        char keypress = waitKey(5);  // This is needed if you want to see anything!

        if (keypress == VK_ESCAPE) {   // Escape Key
            // Quit the program!
            break;
        }

    }//end while
}


int main(int argc, char *argv[])
{
    VideoCapture videoCapture;

    faceDetector = new FaceDetector(shapePredictorFileName);
    // Allow the user to specify a camera number, since not all computers will be the same camera number.
    int cameraNumber = 0;   // Change this if you want to use a different camera device.

    // Get access to the webcam.
    initWebcam(videoCapture, cameraNumber);

    // Try to set the camera resolution. Note that this only works for some cameras on
    // some computers and only for some drivers, so don't rely on it to work!
    videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
    videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

    // Create a GUI window for display on the screen.
    namedWindow(windowName); // Resizable window, might not work on Windows.

    // Run Face Recogintion interactively from the webcam. This function runs until the user quits.
    recognizeAndTrainUsingWebcam(videoCapture);

    return 0;
}
