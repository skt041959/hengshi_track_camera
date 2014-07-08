#include "core_precomp.hpp"

CvMat* cvGetSubRect_d( const CvArr* arr, CvMat* submat, CvRect rect )
{
    CvMat* res = 0;
    CvMat stub, *mat = (CvMat*)arr;

    if( !CV_IS_MAT( mat ))
        mat = cvGetMat( mat, &stub );

    if( !submat )
        CV_Error( CV_StsNullPtr, "" );

    if( (rect.x|rect.y|rect.width|rect.height) < 0 )
        CV_Error( CV_StsBadSize, "" );

    if( rect.x + rect.width > mat->cols ||
        rect.y + rect.height > mat->rows )
        CV_Error( CV_StsBadSize, "" );

    {
        /*
           int* refcount = mat->refcount;

           if( refcount )
           ++*refcount;

           cvDecRefData( submat );
           */
        submat->data.ptr = mat->data.ptr + (size_t)rect.y*mat->step +
            rect.x*CV_ELEM_SIZE(mat->type);
        submat->step = mat->step;
        submat->type = (mat->type & (rect.width < mat->cols ? ~CV_MAT_CONT_FLAG : -1)) |
            (rect.height <= 1 ? CV_MAT_CONT_FLAG : 0);
        submat->rows = rect.height;
        submat->cols = rect.width;
        submat->refcount = 0;
        res = submat;
    }

    return res;
}

