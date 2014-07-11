#include <stdio.h>
#include <pthread.h>

#include "SapClassBasic.h"
#include "SapExUtil.h"

#include "opencv2/video/tracking.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

pthread_mutex_t mutex;
pthread_cond_t grabed = PTHREAD_COND_INITIALIZER;

bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
Mat gray;
int vmin = 10, vmax = 256, smin = 30;

uint32_t paused;

extern int cvCamShift_d( const void* imgProb, CvRect windowIn,
        CvTermCriteria criteria,
        CvConnectedComp* _comp,
        CvBox2D* box );
extern void Canny_d( InputArray _src, OutputArray _dst,
        double low_thresh, double high_thresh,
        int aperture_size, bool L2gradient );

typedef struct tagMY_CONTEXT
{
    uint32_t   height;
    uint32_t   width;
    uint32_t   pixDepth;
    void     *image[1];

}MY_CONTEXT, *PMY_CONTEXT;

static void AcqCallback(SapXferCallbackInfo *pInfo);
void * opencv_main(void *);

void bsize(Mat & src, CvRect & r)
{
    Mat t;
    int i,j;
    int b[4] = {0};

    t = Mat::zeros(1, src.cols, CV_8UC1);
    for(i=0; i<src.rows; i++)
        t += src.row(i);
    for(j=0; j<src.cols; j++)
        if(t.at<uchar>(0, j) == 0u && t.at<uchar>(0, j+1) == 255u)
        { b[0] = j; break; }
    for(j=src.cols-1; j>0; j--)
        if(t.at<uchar>(0, j) == 0u && t.at<uchar>(0, j-1) == 255u)
        { b[1] = j; break; }
    t = Mat::zeros(src.rows, 1, CV_8UC1);
    for(i=0; i<src.cols; i++)
        t += src.col(i);
    for(j=0; j<src.rows; j++)
        if(t.at<uchar>(j, 0) == 0u && t.at<uchar>(j+1, 0) == 255u)
        { b[2] = j; break; }
    for(j=src.rows-1; j>0; j--)
        if(t.at<uchar>(j, 0) == 0u && t.at<uchar>(j-1, 0) == 255u)
        { b[3] = j; break; }
    r.x = b[0];
    r.y = b[2];
    r.width = b[1]-b[0];
    r.height = b[3]-b[2];
}

static void onMouse( int event, int x, int y, int, void* )
{
}

int main(int argc, char * argv[])
{
    SapAcquisition *Acq = NULL;
    SapBuffer *Buffers = NULL;
    SapTransfer *Xfer = NULL;

    MY_CONTEXT context;

    UINT32 pixFormat;
    //char uniqueName[128];
    //char uniqueIndex = 0;
    char acqServerName[CORSERVER_MAX_STRLEN] = "Xcelera-CL_PX4_1";
    char camFilename[MAX_PATH]="A_1000m_10-bits,_Default.ccf";
    char vicFilename[MAX_PATH]="A_1000m_10-bits,_Default.ccf";

    UINT32 acqDeviceNumber = 0;

    //int done = FALSE;
    //int i = 0;
    //char c;

    SapLocation loc(acqServerName, acqDeviceNumber);

    Acq = new SapAcquisition(loc, camFilename, vicFilename);
    Buffers = new SapBuffer(1, Acq);

    Xfer = new SapAcqToBuf(Acq, Buffers, AcqCallback, &context);

    // Create acquisition object
    if (Acq && !*Acq && !Acq->Create())
        goto FreeHandles;

    // Create buffer object
    if (Buffers && !*Buffers && !Buffers->Create())
        goto FreeHandles;

    // Create transfer object
    if (Xfer && !*Xfer && !Xfer->Create())
        goto FreeHandles;

    Acq->GetParameter(CORACQ_PRM_CROP_HEIGHT,&context.height);
    Acq->GetParameter(CORACQ_PRM_CROP_WIDTH,&context.width);
    Acq->GetParameter(CORACQ_PRM_OUTPUT_FORMAT,&pixFormat);
    Acq->GetParameter(CORACQ_PRM_PIXEL_DEPTH, &context.pixDepth); 

    Buffers->GetParameter(CORBUFFER_PRM_ADDRESS,&context.image[0]);

    if( pthread_mutex_init(&mutex, NULL) != 0)
        goto FreeHandles;

    pthread_mutex_unlock( &mutex ); 

    //Mat gray = Mat::zeros(p->width, p->height, CV_16UC1);

    //namedWindow( "CamShift Demo", 0 );

    //while(1)
    //{
    //imshow( "CamShift Demo", gray );
    //}

    pthread_t process_thread;
    paused = 0;
    if( pthread_create(&process_thread, NULL, opencv_main, &context) != 0)
    {
        perror("pthread_create failed\n");
        goto FreeHandles;
    }

    Xfer->Grab();
    printf("Press any key then <enter> to stop grab\n");
    //getchar();
    pthread_join(process_thread, NULL);

    Xfer->Freeze();
    if (!Xfer->Wait(5000))
        printf("Grab cound not stop properly.\n");

FreeHandles:
    printf("Press any key to terminate\n");
    //getchar();

    // Destroy transfer object
    if (Xfer && *Xfer && !Xfer->Destroy())
        return FALSE;

    // Destroy buffer object
    if (Buffers && *Buffers && !Buffers->Destroy())
        return FALSE;

    // Destroy acquisition object
    if (Acq && *Acq && !Acq->Destroy())
        return FALSE;

    // Delete all objects
    if (Xfer)
        delete Xfer;
    if (Buffers)
        delete Buffers;
    if (Acq)
        delete Acq; 

    return 0;
}

void * opencv_main(void * p_context)
{
    char code;
    PMY_CONTEXT p = (PMY_CONTEXT)p_context;

    int hsize = 32;
    float hranges[] = {0,256};
    const float* phranges = hranges;
    char uniqueName[128];
    char filename[128];

    int index = 0;
    int findtarget = 2;
    int begin = 0;

    Mat frame, diff, mask, hist, backproj;
    Mat last, bin1, bin2, eage, gray_out;
    CvRect bord;
    CvRect trackWindow;
    RotatedRect trackBox;
    CvConnectedComp comp;

#ifdef SHOOTING
    strcpy(uniqueName, "CorXXXXXX");
    mkstemp(uniqueName);
#endif

    namedWindow( "back", WINDOW_AUTOSIZE );
    namedWindow( "test", WINDOW_AUTOSIZE );
    namedWindow( "CamShift", WINDOW_AUTOSIZE );

    KalmanFilter KF(4, 2, 0);
    Mat measurement = Mat::zeros(2, 1, CV_32F);
    Mat prediction;
    KF.transitionMatrix = *(Mat_<float>(4, 4) <<\
            1, 0, 1, 0,\
            0, 1, 0, 1,\
            0, 0, 1, 0,\
            0, 0, 0, 1);

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(1));

    while( !paused )
    {
        //printf("////\n");
        pthread_cond_wait( &grabed , &mutex);

        frame = Mat(p->height, p->width, CV_16UC1, p->image[0]);
        frame.convertTo(gray, CV_8UC1, 255.0/1023.0);
        cvtColor(gray, gray_out, CV_GRAY2BGR);

        if( begin )
        {
#ifdef SHOOTING
            if( index<200 )
            {
                sprintf(filename, "%s_img%04d.bmp", uniqueName, index++);
                imwrite(filename, gray);
            }
#else
            if( findtarget )
            {
                if( findtarget>1 )
                {
                    //printf("11111\n");
                    gray.copyTo(last);
                    findtarget--;
                }
                else
                {
                    //printf("22222\n");
                    absdiff(gray, last, diff);
                    GaussianBlur(diff, diff, Size(3, 3), 0);
                    threshold(diff, bin1, 30, 255, CV_THRESH_BINARY);
                    //printf("bin1\n");
                    //imshow("test", bin1);
                    //waitKey(10);
#if 1
                    Mat element = getStructuringElement(MORPH_RECT, Point(10,10));
                    dilate(bin1, bin2, element);
                    erode( bin2, bin1, element);
                    dilate(bin1, bin2, element);
                    //printf("bin2\n");
                    //imshow("test", bin2);
                    //waitKey(1);
                    bsize(bin2, bord);
                    printf("bord %d, %d, %d, %d\n", bord.x, bord.y, bord.width, bord.height);
                    if( bord.width <= 0 || bord.height <= 0 )
                    {
                        //imwrite("error.bmp", bin2);
                        gray.copyTo(last);
                        findtarget = 1;
                        goto staticimg;
                    }

                    Mat roi_gray(gray, bord);
                    eage = Mat::zeros( gray.rows, gray.cols, CV_8UC1);
                    Mat roi_eage(eage, bord);
                    Canny_d(roi_gray, roi_eage, 1000, 4000, 5, 3);

                    vector< vector<Point> > contours;
                    findContours(roi_eage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
                    drawContours(roi_eage, contours, -1, Scalar(255), CV_FILLED);
                    imshow("test", eage);
                    //waitKey(1);
                    calcHist(&roi_gray, 1, 0, roi_eage, hist, 1, &hsize, &phranges);
                    normalize(hist, hist, 0, 255, CV_MINMAX);
                    findtarget--;
                    trackWindow = bord;
                    if( KF.statePost.at<float>(0) == 0 &&
                            KF.statePost.at<float>(1) == 0 &&
                            KF.statePost.at<float>(2) == 0 &&
                            KF.statePost.at<float>(3) == 0)
                    {
                        printf("set KF start state\n");
                        KF.statePost.at<float>(0) = bord.x+bord.width/2.f;
                        KF.statePost.at<float>(1) = bord.y+bord.height/2.f;
                    }
                    goto cam;
                }
            }
            else
            {
                //printf("33333\n");
cam:
                calcBackProject(&gray, 1, 0, hist, backproj, &phranges);
                CvBox2D box;

                CvTermCriteria term;
                term.max_iter = 100;
                term.epsilon = 1;
                term.type = 3;

                CvRect rect = trackWindow;

                //prediction = KF.predict();
                //printf("prediction %f, %f\n", prediction.at<float>(0), prediction.at<float>(1));
                
                //int dx, dy;
                //dx = prediction.at<float>(0) - (trackWindow.x+trackWindow.width/2.f);
                //dy = prediction.at<float>(1) - (trackWindow.y+trackWindow.height/2.f);
                //if(dx < 0)
                //    rect.x = trackWindow.x + dx; 
                //else
                //    rect.width = trackWindow.width + dx; 

                //if(dy < 0)
                //    rect.y = trackWindow.y + dy; 
                //else
                //    rect.height = trackWindow.height + dy;

                //printf("seach   x:%d, y:%d, width:%d, height:%d\n", rect.x, rect.y, rect.x+rect.width, rect.y+rect.height);

                CvMat c_probImage = backproj;
                int ret = cvCamShift_d(&c_probImage, rect, term, &comp, &box);
                unsigned int direction=0;
                int iter = 4;
                //if(dx>0)
                    //direction |= 0x1;
                //if(dy>0)
                    //direction |= 0x2;
                while(ret == -1 && iter)
                {
                    CvRect rect_alt = rect;
                    switch( (direction+iter)%4 )
                    {
                        case 0: rect_alt.x -= rect.width;
                                rect_alt.y -= rect.height;
                                break;
                        case 1: rect_alt.x -= rect.width;
                                rect_alt.y += rect.height;
                                break;
                        case 2: rect_alt.x += rect.width;
                                rect_alt.y -= rect.height;
                                break;
                        case 3: rect_alt.x += rect.width;
                                rect_alt.y += rect.height;
                                break;
                    }
                    printf("lost...search%d  x1:%d, y1:%d, x2:%d, y2:%d\n", iter, rect_alt.x, rect_alt.y, rect_alt.x+rect_alt.width, rect_alt.y+rect_alt.height);
                    ret = cvCamShift_d(&c_probImage, rect, term, &comp, &box);
                    iter--;
                }
                if(ret == -1)
                {
                    printf("retry......\n");
                    gray.copyTo(last);
                    findtarget = 1;
                    goto lost;
                }
                trackBox = RotatedRect(Point2f(box.center), Size2f(box.size), box.angle);
                trackWindow = comp.rect;
                printf("measure x:%d, y:%d, width:%d, height:%d\n", trackWindow.x, trackWindow.y, trackWindow.width, trackWindow.height);

                measurement.at<float>(0) = trackBox.center.x;
                measurement.at<float>(1) = trackBox.center.y;
                //KF.correct(measurement);
                ellipse(gray_out, trackBox, Scalar(0, 0, 255), 1, CV_AA);
lost:
                imshow("back", backproj);
#endif
            }
#endif
        }
staticimg:
        imshow("CamShift", gray_out);
        code = (char)waitKey(10);

        if(code=='q')
            paused = 1;

        if(code=='b')
            begin = 1;

        if(code=='r')
            findtarget = 2;

        pthread_mutex_unlock( &mutex );
    }
exit:
    pthread_exit(0);
}

static void AcqCallback(SapXferCallbackInfo *pInfo)
{
    int ret;
    if ( pthread_mutex_trylock(&mutex) == 0)
    {
        pthread_cond_signal( &grabed );
        ret = pthread_mutex_unlock( &mutex ); 
    }
}

