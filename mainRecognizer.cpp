
const char *shapePredictorFileName = "/home/davy/shape_predictor_68_face_landmarks.dat";
const char *facerecAlgorithm = "FaceRecognizer.Eigenfaces";

// Sets how confident the Face Verification algorithm should be to decide if it is an unknown person or a known person.
// A value roughly around 0.5 seems OK for Eigenfaces or 0.7 for Fisherfaces, but you may want to adjust it for your
// conditions, and if you use a different Face Recognition algorithm.
// Note that a higher threshold value means accepting more faces as known people,
// whereas lower values mean more faces will be classified as "unknown".
const float UNKNOWN_PERSON_THRESHOLD = 0.5f;

// Set the desired face dimensions. Note that "getPreprocessedFace()" will return a square face.
const int faceWidth = 70;
const int faceHeight = faceWidth;

// Try to set the camera resolution. Note that this only works for some cameras on
// some computers and only for some drivers, so don't rely on it to work!
const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;

// Parameters controlling how often to keep new faces when collecting them. Otherwise, the training set could look to similar to each other!
const double CHANGE_IN_IMAGE_FOR_COLLECTION = 0.3;      // How much the facial image should change before collecting a new face photo for training.
const double CHANGE_IN_SECONDS_FOR_COLLECTION = 1.0;       // How much time must pass before collecting a new face photo for training.

const char *windowName = "WebcamFaceRec";   // Name shown in the GUI window.
const int BORDER = 8;  // Border between GUI elements to the edge of the image.

// Set to true if you want to see many windows created, showing various debug info. Set to 0 otherwise.
bool m_debug = false;

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
#include "FaceDetector.h"     // Easily preprocess face images, for face recognition.
#include "Recognition.h"     // Train the face recognition system and recognize a person from an image.


using namespace cv;
//using namespace dlib;
using namespace std;


#if !defined VK_ESCAPE
#define VK_ESCAPE 0x1B      // Escape character (27)
#endif


// Running mode for the Webcam-based interactive GUI program.
enum MODES {MODE_STARTUP=0, MODE_RECOGNITION, MODE_END};
const char* MODE_NAMES[] = {"Startup", "Recognition","ERROR!"};
MODES m_mode = MODE_STARTUP;

Recognition *recognizer;
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
void recognizeUsingWebcam(VideoCapture &videoCapture)
{

    m_mode = MODE_RECOGNITION;


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

        // Run the face recognition system on the camera image. It will draw some things onto the given image, so make sure it is not read-only memory!
        int identity = -1;

        // Find a face and preprocess it to have a standard size and contrast & brightness.
        Rect faceRect;  // Position of detected face.
        Point leftEye, rightEye;    // Position of the detected eyes.
        bool gotFaceAndEyes = faceDetector->detectFaceAndEyes(displayedFrame,&faceRect, &leftEye,&rightEye);
        Mat preprocessedFace;
        if(gotFaceAndEyes)
        {
            Mat face = displayedFrame(faceRect);
            preprocessedFace = faceDetector->getPreprocessedFace(face, faceWidth, leftEye, rightEye);
        }

        // Draw an anti-aliased rectangle around the detected face.
        if (faceRect.width > 0) {
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


       if (m_mode == MODE_RECOGNITION) {
            if (gotFaceAndEyes ) {

                // Generate a face approximation by back-projecting the eigenvectors & eigenvalues.
                Mat reconstructedFace;
                reconstructedFace = recognizer->reconstructFace(preprocessedFace);

                // Verify whether the reconstructed face looks like the preprocessed face, otherwise it is probably an unknown person.
                double similarity = recognizer->getSimilarity(preprocessedFace, reconstructedFace);

                string outputStr;
                if (similarity < UNKNOWN_PERSON_THRESHOLD) {
                    // Identify who the person is in the preprocessed face image.
                    identity = recognizer->predict(preprocessedFace);
                    outputStr = "" + identity;
                }
                else {
                    // Since the confidence is low, assume it is an unknown person.
                    outputStr = "Unknown";
                }
                cout << "Identity: " << outputStr << ". Similarity: " << similarity << endl;

                // Show the confidence rating for the recognition in the mid-top of the display.
                int cx = (displayedFrame.cols - faceWidth) / 2;
                Point ptBottomRight = Point(cx - 5, BORDER + faceHeight);
                Point ptTopLeft = Point(cx - 15, BORDER);
                // Draw a gray line showing the threshold for an "unknown" person.
                Point ptThreshold = Point(ptTopLeft.x, ptBottomRight.y - (1.0 - UNKNOWN_PERSON_THRESHOLD) * faceHeight);
                rectangle(displayedFrame, ptThreshold, Point(ptBottomRight.x, ptThreshold.y), CV_RGB(200,200,200), 1, CV_AA);
                // Crop the confidence rating between 0.0 to 1.0, to show in the bar.
                double confidenceRatio = 1.0 - min(max(similarity, 0.0), 1.0);
                Point ptConfidence = Point(ptTopLeft.x, ptBottomRight.y - confidenceRatio * faceHeight);
                // Show the light-blue confidence bar.
                rectangle(displayedFrame, ptConfidence, ptBottomRight, CV_RGB(0,255,255), CV_FILLED, CV_AA);
                // Show the gray border of the bar.
                rectangle(displayedFrame, ptTopLeft, ptBottomRight, CV_RGB(200,200,200), 1, CV_AA);

                //Display picture of person
            }
        }
        else {
            cerr << "ERROR: Invalid run mode " << m_mode << endl;
            exit(1);
        }

        // Show the camera frame on the screen.
        imshow(windowName, displayedFrame);

        // IMPORTANT: Wait for atleast 20 milliseconds, so that the image can be displayed on the screen!
        // Also checks if a key was pressed in the GUI window. Note that it should be a "char" to support Linux.
        char keypress = waitKey(20);  // This is needed if you want to see anything!

        if (keypress == VK_ESCAPE) {   // Escape Key
            // Quit the program!
            break;
        }

    }//end while
}


int main(int argc, char *argv[])
{

    VideoCapture videoCapture;

    recognizer = new Recognition(facerecAlgorithm);
    faceDetector = new FaceDetector(shapePredictorFileName);

    cout << "Hit 'Escape' in the GUI window to quit." << endl;

    // Allow the user to specify a camera number, since not all computers will be the same camera number.
    int cameraNumber = 0;   // Change this if you want to use a different camera device.
    if (argc > 1) {
        cameraNumber = atoi(argv[1]);
    }

    // Get access to the webcam.
    initWebcam(videoCapture, cameraNumber);

    // Try to set the camera resolution. Note that this only works for some cameras on
    // some computers and only for some drivers, so don't rely on it to work!
    videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
    videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

    // Create a GUI window for display on the screen.
    namedWindow(windowName); // Resizable window, might not work on Windows.

    // Run Face Recogintion interactively from the webcam. This function runs until the user quits.
    recognizeUsingWebcam(videoCapture);

    return 0;
}
