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

typedef struct tagMY_CONTEXT
{
    uint32_t   height;
    uint32_t   width;
    uint32_t   pixDepth;
    void     *image[1];

}MY_CONTEXT, *PMY_CONTEXT;

static void AcqCallback(SapXferCallbackInfo *pInfo);
void * opencv_main(void *);

void bsize(Mat & src, Rect & r)
{
    Mat t;
    int i,j;
    int b[4];
    t = Mat::zeros(1, src.cols, CV_8UC1);
    for(i=0; i<src.rows; i++)
    {
        t += src.row(i);
    }
    for(j=0; j<src.cols; j++)
    {
        //printf("%d\n",t.at<uchar>(0, j));
        if(t.at<uchar>(0, j) == 0u && t.at<uchar>(0, j+1) == 255u)
        {
            //printf("===\n");
            b[0] = j;
        }
        if(t.at<uchar>(0, j) == 255u && t.at<uchar>(0, j+1) == 0u)
        {
            //printf("===\n");
            b[1] = j;
        }
    }
    //printf("---\n");
    t = Mat::zeros(src.rows, 1, CV_8UC1);
    for(i=0; i<src.cols; i++)
    {
        t += src.col(i);
    }
    for(j=0; j<src.rows; j++)
    {
        //printf("%d\n",t.at<uchar>(0, j));
        if(t.at<uchar>(j, 0) == 0u && t.at<uchar>(j+1, 0) == 255u)
        {
            //printf("===\n");
            b[2] = j;
        }
        if(t.at<uchar>(j, 0) == 255u && t.at<uchar>(j+1, 0) == 0u)
        {
            //printf("===\n");
            b[3] = j;
        }
    }
    r.x = b[0];
    r.y = b[2];
    r.width = b[1]-b[0];
    r.height = b[3]-b[2];
}

static void onMouse( int event, int x, int y, int, void* )
{
    if( selectObject )
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);

        selection &= Rect(0, 0, gray.cols, gray.rows);
    }

    switch( event )
    {
        case CV_EVENT_LBUTTONDOWN:
            origin = Point(x,y);
            selection = Rect(x,y,0,0);
            selectObject = true;
            break;
        case CV_EVENT_LBUTTONUP:
            selectObject = false;
            if( selection.width > 0 && selection.height > 0 )
                trackObject = -1;
            break;
    }
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

    Rect trackWindow;
    int hsize = 16;
    int index = 0;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    char uniqueName[128];
    char filename[128];

    int findtarget = 2;
    int begin = 0;

    namedWindow("CamShift", WINDOW_AUTOSIZE );
    namedWindow("back", WINDOW_AUTOSIZE );
    namedWindow("test", WINDOW_AUTOSIZE );

    Mat frame, diff, mask, hist, backproj;
    Mat last, bin1, bin2, eage, tmp;
    Rect bord;

    strcpy(uniqueName, "CorXXXXXX");
    mkstemp(uniqueName);

    while( !paused )
    {
        printf("////\n");
        pthread_cond_wait( &grabed , &mutex);

        frame = Mat(p->height, p->width, CV_16UC1, p->image[0]);
        frame.convertTo(gray, CV_8UC1, 255.0/1023.0);

        if( begin )
        {
            if(index<200)
            {
                sprintf(filename, "%s_img%04d.bmp", uniqueName, index++);
                imwrite(filename, gray);
            }
            /*
            if( findtarget )
            {
                if( findtarget>1 )
                {
                    printf("11111\n");
                    gray.copyTo(last);
                    findtarget--;
                }
                else
                {
                    printf("22222\n");
                    absdiff(gray, last, diff);
                    GaussianBlur(diff, diff, Size(3, 3), 0);
                    threshold(diff, bin1, 5, 255, CV_THRESH_BINARY);

                    Mat element = getStructuringElement(MORPH_RECT, Point(10,10));
                    dilate(bin1, bin2, element);

                    bsize(bin2, bord);
                    printf("%d, %d, %d, %d\n", bord.x, bord.y, bord.width, bord.height);

                    eage = Mat::zeros( gray.rows, gray.cols, CV_8UC1);

                    Mat roi_gray(gray, bord);
                    Mat roi_eage(eage, bord);
                    Canny(roi_gray, roi_eage, 1000, 4000, 5);
                    vector< vector<Point> > contours;
                    findContours(roi_eage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
                    drawContours(roi_eage, contours, -1, Scalar(255), CV_FILLED);

                    calcHist(&roi_gray, 1, 0, roi_eage, hist, 1, &hsize, &phranges);

                    normalize(hist, hist, 0, 255, CV_MINMAX);
                    imshow("test", diff);
                    //waitKey(-1);
                    findtarget--;
                    trackWindow = bord;
                }
            }
            else
            {
                //sprintf(filename, "CorUR1C8s_img%04d.bmp", index++);
                //gray = imread(filename, IMREAD_GRAYSCALE);
                printf("33333\n");

                calcBackProject(&gray, 1, 0, hist, backproj, &phranges);
                RotatedRect trackBox = CamShift(backproj, trackWindow,
                        TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
                ellipse(gray, trackBox, Scalar(255), 1, CV_AA);
                imshow("back", backproj);
            }
            */
        }

        imshow("CamShift", gray);
        code = (char)waitKey(10);
        if(code=='q')
            paused = 1;

        if(code=='b')
            begin = 1;

        if(code=='r')
            findtarget = 2;

        pthread_mutex_unlock( &mutex );
    }

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

