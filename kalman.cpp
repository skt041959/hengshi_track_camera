#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>

#define drawCross( center, color, d )\
    line( img, Point( center.x - d, center.y - d ),\
    Point( center.x + d, center.y + d ), color, 1, CV_AA, 0); \
    line( img, Point( center.x + d, center.y - d ),\
    Point( center.x - d, center.y + d ), color, 1, CV_AA, 0 )

using namespace cv;

static inline Point calcPoint(Point2f center, double R, double angle)
{
    return center + Point2f((float)cos(angle), (float)-sin(angle))*(float)R;
}

static void help()
{
    printf( "\nExamle of c calls to OpenCV's Kalman filter.\n"
"   Tracking of rotating point.\n"
"   Rotation speed is constant.\n"
"   Both state and measurements vectors are 1D (a point angle),\n"
"   Measurement is the real point angle + gaussian noise.\n"
"   The real and the estimated points are connected with yellow line segment,\n"
"   the real and the measured points are connected with red line segment.\n"
"   (if Kalman filter works correctly,\n"
"    the yellow segment should be shorter than the red one).\n"
            "\n"
"   Pressing any key (except ESC) will reset the tracking with a different speed.\n"
"   Pressing ESC will stop the program.\n"
            );
}

int main(int, char**)
{
    KalmanFilter KF(4, 2, 0);
    Mat state(4, 1, CV_32F); /* (x, y) */
    //Mat processNoise(2, 1, CV_32F);
    Mat measurement = Mat::zeros(2, 1, CV_32F);
    char code = (char)-1;

    //for(;;)
    {
        //randn( state, Scalar::all(0), Scalar::all(0.1) );
        KF.transitionMatrix = *(Mat_<float>(4, 4) << 1, 0, 1, 0,\
                0, 1, 0, 1,\
                0, 0, 1, 0,\
                0, 0, 0, 1);

        setIdentity(KF.measurementMatrix);
        setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
        setIdentity(KF.errorCovPost, Scalar::all(1));

        //randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));
        Point2f d1(791, 308);
        Point2f d2(792, 298);
        Point2f d3(793, 286);
        Point2f d4(794, 273);

        Mat prediction;

        //prediction = KF.predict();
        //printf("%f, %f, %f, %f\n", prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2), prediction.at<float>(3));
        //measurement.at<float>(0) = d1.x;
        //measurement.at<float>(1) = d1.y;
        //KF.correct(measurement);
        KF.statePost.at<float>(0) = d1.x;
        KF.statePost.at<float>(1) = d1.y;
        KF.statePost.at<float>(2) = 0;
        KF.statePost.at<float>(3) = 0;
        printf("%f, %f, %f, %f\n", KF.statePost.at<float>(0), KF.statePost.at<float>(1), KF.statePost.at<float>(2), KF.statePost.at<float>(3));

        prediction = KF.predict();
        printf("%f, %f, %f, %f\n", prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2), prediction.at<float>(3));
        measurement.at<float>(0) = d2.x;
        measurement.at<float>(1) = d2.y;
        KF.correct(measurement);

        prediction = KF.predict();
        printf("%f, %f, %f, %f\n", prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2), prediction.at<float>(3));
        measurement.at<float>(0) = d3.x;
        measurement.at<float>(1) = d3.y;
        KF.correct(measurement);

        prediction = KF.predict();
        printf("%f, %f, %f, %f\n", prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2), prediction.at<float>(3));
        measurement.at<float>(0) = d4.x;
        measurement.at<float>(1) = d4.y;
        KF.correct(measurement);

        prediction = KF.predict();
        printf("%f, %f, %f, %f\n", prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2), prediction.at<float>(3));
    }

    return 0;
}
