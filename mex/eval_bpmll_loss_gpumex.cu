#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"

void __global__ compute_deriv(double const * const T,
						double const * const P,
						double * const delta,
						int const nDim, int const nExamples)
{
	int const x = blockDim.x * blockIdx.x + threadIdx.x;
	if(x >= nDim*nExamples) return;

	unsigned int m = x/nDim;
	unsigned int j = x%nDim;

	unsigned int k,l;
	double delta_sum = 0.0;
	unsigned int n_pos_labels = 0;

	if(T[j+m*nDim]== 1) {			// if label i for instance j is positive
		for(l=0;l<nDim;l++) if(T[l+m*nDim]==-1) delta_sum += exp(-P[j+m*nDim]+P[l+m*nDim]);
		delta_sum = -delta_sum;
	} else if(T[j+m*nDim]==-1) {	// if label l is negative
		for(k=0;k<nDim;k++) if(T[k+m*nDim]== 1) delta_sum += exp(-P[k+m*nDim]+P[j+m*nDim]);
	}
	for(k=0;k<nDim;k++) if(T[k+m*nDim]==1) n_pos_labels++;
	delta[j+m*nDim] = 1.0/(n_pos_labels*(nDim-n_pos_labels))*delta_sum;
}

void mexFunction(int nlhs, mxArray *plhs[], /* output variables */
				int nrhs, const mxArray *prhs[])
{
	mxGPUArray const *T;
	mxGPUArray const *P;
	mxGPUArray *delta;
	double const *d_T, *d_P;
	double *d_delta;

	/* Choose a reasonably sized number of threads for the block. */
	int const threadsPerBlock = 256;
	int blocksPerGrid;
	int N;

	/* Initialize the GPU API */
	mxInitGPU();

	if ((nrhs!=2) || !(mxIsGPUArray(prhs[0])) || !(mxIsGPUArray(prhs[1]))) {
		mexErrMsgIdAndTxt("parallel:gpu:BPMLL:InvalidInput", "Invalid input to MEX file.");
	}

	T = mxGPUCreateFromMxArray(prhs[0]);	/* targets */
	P = mxGPUCreateFromMxArray(prhs[1]);	/* predictions */

	/* Verify that P and T really are double arrays before extracting the pointer. */
	if ((mxGPUGetClassID(T) != mxDOUBLE_CLASS) || (mxGPUGetClassID(P) != mxDOUBLE_CLASS)) {
		mexErrMsgIdAndTxt("parallel:gpu:BPMLL:InvalidInput", "Invalid input to MEX file.");
	}

	d_T = (double const *)(mxGPUGetDataReadOnly(T));
	d_P = (double const *)(mxGPUGetDataReadOnly(P));

	mwSize const *delta_dims = mxGPUGetDimensions(P);
	int const nDim = delta_dims[0];
	int const nExamples = delta_dims[2];

	/* Create a GPUArray to hold the result and get its underlying pointer. */

	delta = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(P),
								delta_dims,
								mxGPUGetClassID(P),
								mxGPUGetComplexity(P),
								MX_GPU_DO_NOT_INITIALIZE);
	d_delta = (double *)(mxGPUGetData(delta));

	N = (int)(mxGPUGetNumberOfElements(P));
	blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;
	compute_deriv<<<blocksPerGrid, threadsPerBlock>>>(d_T,d_P,d_delta,nDim,nExamples);

	/* Wrap the result up as a MATLAB gpuArray for return. */
	plhs[0] = mxGPUCreateMxArrayOnGPU(delta);

	mxGPUDestroyGPUArray(T);
	mxGPUDestroyGPUArray(P);
	mxGPUDestroyGPUArray(delta);

	mxFree((void*)delta_dims);

	return;
}
