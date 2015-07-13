//
// Created by davy on 3/12/15.
//




#include "FaceDetector.h"

FaceDetector::FaceDetector(const char *shapePredictorFileName) {

    detector = dlib::get_frontal_face_detector();
    cout << "Loaded the Frontal Face Detector from Dlib" << endl;
    dlib::deserialize(shapePredictorFileName) >> shapePredictor;
    cout << "Loaded the shapePredictor from Dlib" << endl;


}

void FaceDetector::detectBothEyes(const Mat &face, Point &leftEye, Point &rightEye) {

    dlib::cv_image<dlib::bgr_pixel> cv(face);
    dlib::array2d<dlib::rgb_pixel> img;
    dlib::assign_image(img,cv);

    dlib::full_object_detection shape = shapePredictor(img,dlib::rectangle(img.nc(), img.nr()));
    int k; dlib::point meanl, meanr;
    for(k=36; k<42; k++)
        meanl += shape.part(k);
    meanl /= 6;
    leftEye.x = meanl.x();
    leftEye.y = meanl.y();

    for(k=42; k<48; k++)
        meanr += shape.part(k);
    meanr /= 6;
    rightEye.x = meanr.x();
    rightEye.y = meanr.y();

}

void FaceDetector::equalizeLeftAndRightHalves(Mat &faceImg) {

    // It is common that there is stronger light from one half of the face than the other. In that case,
    // if you simply did histogram equalization on the whole face then it would make one half dark and
    // one half bright. So we will do histogram equalization separately on each face half, so they will
    // both look similar on average. But this would cause a sharp edge in the middle of the face, because
    // the left half and right half would be suddenly different. So we also histogram equalize the whole
    // image, and in the middle part we blend the 3 images together for a smooth brightness transition.

    int w = faceImg.cols;
    int h = faceImg.rows;

    // 1) First, equalize the whole face.
    Mat wholeFace;
    equalizeHist(faceImg, wholeFace);

    // 2) Equalize the left half and the right half of the face separately.
    int midX = w/2;
    Mat leftSide = faceImg(Rect(0,0, midX,h));
    Mat rightSide = faceImg(Rect(midX,0, w-midX,h));
    equalizeHist(leftSide, leftSide);
    equalizeHist(rightSide, rightSide);

    // 3) Combine the left half and right half and whole face together, so that it has a smooth transition.
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            int v;
            if (x < w/4) {          // Left 25%: just use the left face.
                v = leftSide.at<uchar>(y,x);
            }
            else if (x < w*2/4) {   // Mid-left 25%: blend the left face & whole face.
                int lv = leftSide.at<uchar>(y,x);
                int wv = wholeFace.at<uchar>(y,x);
                // Blend more of the whole face as it moves further right along the face.
                float f = (x - w*1/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * lv + (f) * wv);
            }
            else if (x < w*3/4) {   // Mid-right 25%: blend the right face & whole face.
                int rv = rightSide.at<uchar>(y,x-midX);
                int wv = wholeFace.at<uchar>(y,x);
                // Blend more of the right-side face as it moves further right along the face.
                float f = (x - w*2/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * wv + (f) * rv);
            }
            else {                  // Right 25%: just use the right face.
                v = rightSide.at<uchar>(y,x-midX);
            }
            faceImg.at<uchar>(y,x) = v;
        }// end x loop
    }//end y loop

}

Mat FaceDetector::getPreprocessedFace(Mat &faceImg, int desiredFaceWidth, Point leftEye, Point rightEye) {
    // Use square faces.
    int desiredFaceHeight = desiredFaceWidth;


    // If the input image is not grayscale, then convert the BGR or BGRA color image to grayscale.
    Mat gray;
    if (faceImg.channels() == 3) {
        cvtColor(faceImg, gray, CV_BGR2GRAY);
    }
    else if (faceImg.channels() == 4) {
        cvtColor(faceImg, gray, CV_BGRA2GRAY);
    }
    else {
        // Access the input image directly, since it is already grayscale.
        gray = faceImg;
    }

    // Make the face image the same size as the training images.

    // Since we found both eyes, lets rotate & scale & translate the face so that the 2 eyes
    // line up perfectly with ideal eye positions. This makes sure that eyes will be horizontal,
    // and not too far left or right of the face, etc.

    // Get the center between the 2 eyes.
    Point2f eyesCenter = Point2f((leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f);
    // Get the angle between the 2 eyes.
    double dy = (rightEye.y - leftEye.y);
    double dx = (rightEye.x - leftEye.x);
    double len = sqrt(dx * dx + dy * dy);
    double angle = atan2(dy, dx) * 180.0 / CV_PI; // Convert from radians to degrees.

    // Hand measurements shown that the left eye center should ideally be at roughly (0.19, 0.14) of a scaled face image.
    const double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);
    // Get the amount we need to scale the image to be the desired fixed size we want.
    double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * desiredFaceWidth;
    double scale = desiredLen / len;
    // Get the transformation matrix for rotating and scaling the face to the desired angle & size.
    Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
    // Shift the center of the eyes to be the desired center between the eyes.
    rot_mat.at<double>(0, 2) += desiredFaceWidth * 0.5f - eyesCenter.x;
    rot_mat.at<double>(1, 2) += desiredFaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.y;

    // Rotate and scale and translate the image to the desired angle & size & position!
    // Note that we use 'w' for the height instead of 'h', because the input face has 1:1 aspect ratio.
    Mat warped = Mat(desiredFaceHeight, desiredFaceWidth, CV_8U, Scalar(128)); // Clear the output image to a default grey.
    warpAffine(gray, warped, rot_mat, warped.size());
    //imshow("warped", warped);

    // Do it seperately for the left and right sides of the face.
    equalizeLeftAndRightHalves(warped);


    // Use the "Bilateral Filter" to reduce pixel noise by smoothing the image, but keeping the sharp edges in the face.
    Mat filtered = Mat(warped.size(), CV_8U);
    bilateralFilter(warped, filtered, 0, 20.0, 2.0);
    //imshow("filtered", filtered);

    // Filter out the corners of the face, since we mainly just care about the middle parts.
    // Draw a filled ellipse in the middle of the face-sized image.
    Mat mask = Mat(warped.size(), CV_8U, Scalar(0)); // Start with an empty mask.
    Point faceCenter = Point(desiredFaceWidth / 2, cvRound(desiredFaceHeight * FACE_ELLIPSE_CY));
    Size size = Size(cvRound(desiredFaceWidth * FACE_ELLIPSE_W), cvRound(desiredFaceHeight * FACE_ELLIPSE_H));
    ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
    //imshow("mask", mask);

    //imshow("equalized", warped);
    // Use the mask, to remove outside pixels.
    Mat dstImg = Mat(warped.size(), CV_8U, Scalar(128)); // Clear the output image to a default gray.

    // Apply the elliptical mask on the face.
    filtered.copyTo(dstImg, mask);  // Copies non-masked pixels from filtered to dstImg.
    //imshow("dstImg", dstImg);

    return dstImg;
}

bool FaceDetector::detectFaceAndEyes(Mat &srcImg, Rect *storeFaceRect, Point *storeLeftEye, Point *storeRightEye) {

    // Mark the detected face region as invalid, in case they aren't detected.
    if (storeFaceRect)
        storeFaceRect->width = -1;

    // Find the largest face.
    Rect faceRect;

    dlib::cv_image<dlib::bgr_pixel> cv(srcImg);
    dlib::array2d<dlib::rgb_pixel> img;
    dlib::assign_image(img,cv);

    std::vector<dlib::rectangle> dets = detector(img);
    if(dets.size() > 0)
    {
        faceRect =  Rect(dets[0].left(), dets[0].top(), dets[0].width(),
                dets[0].height())  &Rect(0, 0, srcImg.size().width, srcImg.size().height);

        // Give the face rect to the caller if desired.
        if (storeFaceRect)
            *storeFaceRect = faceRect;

        Mat faceImg = srcImg(faceRect);    // Get the detected face image.

        // Search for the 2 eyes at the full resolution, since eye detection needs max resolution possible!
        Point leftEye, rightEye;
        detectBothEyes(faceImg, leftEye, rightEye);

        // Check if both eyes were detected.
        if (leftEye.x >= 0 && rightEye.x >= 0) {
            *storeLeftEye = leftEye;
            *storeRightEye = rightEye;

            return true;

        }
    }
    return false;
}
