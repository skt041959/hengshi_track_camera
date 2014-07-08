#ifndef _CV_GEOM_H_
#define _CV_GEOM_H_

/* Finds distance between two points */
CV_INLINE  float  icvDistanceL2_32f( CvPoint2D32f pt1, CvPoint2D32f pt2 )
{
    float dx = pt2.x - pt1.x;
    float dy = pt2.y - pt1.y;

    return std::sqrt( dx*dx + dy*dy );
}


int  icvIntersectLines( double x1, double dx1, double y1, double dy1,
                        double x2, double dx2, double y2, double dy2,
                        double* t2 );


void icvCreateCenterNormalLine( CvSubdiv2DEdge edge, double* a, double* b, double* c );

void icvIntersectLines3( double* a0, double* b0, double* c0,
                         double* a1, double* b1, double* c1,
                         CvPoint2D32f* point );


/* curvature: 0 - 1-curvature, 1 - k-cosine curvature. */
CvSeq* icvApproximateChainTC89( CvChain* chain, int header_size, CvMemStorage* storage, int method );

#endif /*_IPCVGEOM_H_*/

/* End of file. */
