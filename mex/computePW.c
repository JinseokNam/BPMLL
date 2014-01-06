#include <math.h>
#include "mex.h"

void compute(double *T, double *P, int nDim, int nExamples, double *error, double *delta)
{
    int n_pos_labels, *pos_label_idx;
    unsigned int i,m,k,l;
    double err_sum;
    
    for(m=0;m<nExamples;m++) {
        err_sum = 0.0;
        n_pos_labels = 0;

        for(k=0;k<nDim;k++) if(T[k+nDim*m] == 1)    n_pos_labels += 1;
        pos_label_idx = malloc(n_pos_labels*sizeof(int));
        l=0;
        for(k=0;k<nDim;k++) {
            if(T[k+nDim*m] != 1) continue;
            pos_label_idx[l++] = k;
        }

        for(k=0;k<nDim;k++) {
            if(T[k+nDim*m] != 1) {  /* label k is irrelevant */
                /* compute gradient for label k */
                for(i=0;i<n_pos_labels;i++) {
                    delta[k+nDim*m] += exp(-P[pos_label_idx[i]+nDim*m]+P[k+nDim*m]);
                }
            } else {                /* label k is relevant */
                for(l=0;l<nDim;l++) {
                    if(T[l+nDim*m] == 1)    continue;
                    err_sum += exp(-P[k+nDim*m] + P[l+nDim*m]);
                    delta[k+nDim*m] += exp(-P[k+nDim*m]+P[l+nDim*m]);
                }
                delta[k+nDim*m] = -delta[k+nDim*m];
            }
            delta[k+nDim*m] = (delta[k+nDim*m]/(n_pos_labels*(nDim-n_pos_labels)));
        }
        error[m] = err_sum/(n_pos_labels*(nDim-n_pos_labels));
        free(pos_label_idx);
    }
}

void mexFunction(int nlhs, mxArray *plhs[], /* output variables */
                int nrhs, const mxArray *prhs[])
{

    double *T, *P, *error, *delta;
    unsigned int i;
    int nExamples, nDim;

    if (nrhs < 1 || nrhs > 3)
        mexErrMsgTxt("Invalid input arguments");
    else if (nlhs != 2)
        mexErrMsgTxt("Invalid output arguments");

    nExamples = mxGetN(prhs[0]);
    nDim = mxGetM(prhs[0]);

    if(nExamples != mxGetN(prhs[1]) || nDim != mxGetM(prhs[1]))
        mexErrMsgTxt("Dimension mismatch");

    T = mxGetPr(prhs[0]);   /* targets */
    P = mxGetPr(prhs[1]);   /* prediction */

    plhs[0] = mxCreateDoubleMatrix(1,nExamples,mxREAL); 
    error = mxGetPr(plhs[0]);

    plhs[1] = mxCreateDoubleMatrix(nDim,nExamples,mxREAL);
    delta = mxGetPr(plhs[1]);
    
    compute(T,P,nDim,nExamples,error,delta);    

    return;
}
