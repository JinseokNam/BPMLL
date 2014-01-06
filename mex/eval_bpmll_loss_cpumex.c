#include <math.h>
#include "mex.h"
#include <omp.h>

void compute_deriv_per_example(double const * const T,
                double  const * const P,
                const int nDim,
                const int m, 
                double * const delta_m)
{
    unsigned int k,l,j;
    double delta_sum;
    unsigned int n_pos_labels;

    for(j=0;j<nDim;j++) {
        delta_sum = 0.0;
        n_pos_labels = 0;
        if(T[j+m*nDim]== 1) {           
            for(l=0;l<nDim;l++) if(T[l+m*nDim]==-1) delta_sum += exp(-P[j+m*nDim]+P[l+m*nDim]);
            delta_sum = -delta_sum;
        } else if(T[j+m*nDim]==-1) {    
            for(k=0;k<nDim;k++) if(T[k+m*nDim]== 1) delta_sum += exp(-P[k+m*nDim]+P[j+m*nDim]);
        }
        for(k=0;k<nDim;k++) if(T[k+m*nDim]==1) n_pos_labels++;
        delta_m[j] = 1.0/(n_pos_labels*(nDim-n_pos_labels))*delta_sum;
    }
}

void compute_deriv(double const * T,
                double const *P,
                int nDim,
                int nExamples,
                double *delta)
{
    unsigned int m;

    #pragma omp parallel for default(none) shared(nExamples,P,T,nDim,delta) private(m)
    for(m=0;m<nExamples;m++) {
        compute_deriv_per_example(T,P,nDim,m,&delta[m*nDim]);
    }
}


void mexFunction(int nlhs, mxArray *plhs[], /* output variables */
                int nrhs, const mxArray *prhs[])
{

    double *T, *P;
    double *delta;
    unsigned int i;
    int nExamples, nDim;

    if (nrhs < 2 || nrhs > 3)
        mexErrMsgTxt("Invalid input arguments");

    nExamples = mxGetN(prhs[0]);
    nDim = mxGetM(prhs[0]);

    if(nExamples != mxGetN(prhs[1]) || nDim != mxGetM(prhs[1]))
        mexErrMsgTxt("Dimension mismatch");

    T = mxGetPr(prhs[0]);   /* targets */
    P = mxGetPr(prhs[1]);   /* predictions */

    plhs[0] = mxCreateDoubleMatrix(nDim,nExamples,mxREAL);
    delta = mxGetPr(plhs[0]);
    
    compute_deriv(T,P,nDim,nExamples,delta);    

    return;
}
