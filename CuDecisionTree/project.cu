#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "timer.h"
#include "cuda_utils.h"

#include <thrust/sort.h>

typedef float dtype;

#define N_ (8 * 8 * 8)
#define MAX_THREADS 256
#define MAX_BLOCKS 64

#define MIN(x,y) ((x < y) ? x : y)



float *d_attr_values, *d_output;
int * d_target_values;
float * data_temp;

// NVIDIA's
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
__device__ float selection_sort( float * data_mat, int left, int right )
{

    float max = 0;
    float min = 100000;
    for(int i = left; i<right; i++) {
        if(data_mat[i]>max)
            max = data_mat[i];
        if(data_mat[i]<min)
            min = data_mat[i];
    }
    return (max+min)/2;


}


__global__ void
kernel5(float * attrValues, int *targetValues, float *g_odata, unsigned int numSamples, unsigned int numClasses, int numAttributes)
{
  /* Shared variable within each block */
  __shared__  float scratch[MAX_THREADS];
  __shared__  float sum[MAX_THREADS];
  /* Finding block id */
  unsigned int bid = gridDim.x * blockIdx.y + blockIdx.x;
  /* Number of threads within each block */
  unsigned int numberThreads = blockDim.x;
  /* Attributes per block */
  unsigned int blockOffset = bid*numSamples;
  if(threadIdx.x==0 && bid<numAttributes) {
    /* Initialize Shared Memory */
    for(int k = 0; k<3*numClasses; k++) {
        scratch[k] = 0;
    }
    for(int k = 0; k<MAX_THREADS ; k++) {
        sum[k] = 0;
    }
    //Memory to store threshold 
    scratch[3*numClasses] = 0;
  }
  __syncthreads ();
 

  /* Reduction */
  unsigned int offset = blockOffset+threadIdx.x;
  while(offset < blockOffset+numSamples && bid<numAttributes) {
    sum[threadIdx.x] += attrValues[offset];
    offset += numberThreads;     
  }
  __syncthreads ();


  /* Find global max and min */
  if(threadIdx.x==0 && bid<numAttributes){
    float global_sum = 0;
    for(unsigned int m = 0; m<MAX_THREADS; m++)
        global_sum += sum[m];
    scratch[3*numClasses] = global_sum/numSamples;
  }
  __syncthreads ();


  offset = blockOffset+threadIdx.x;
  while(offset < blockOffset+numSamples && bid<numAttributes) {
    int temp = (int)targetValues[offset-blockOffset];
    atomicAdd(scratch+temp, 1);
    float thresh = scratch[3*numClasses];
    if(attrValues[offset] <= thresh)
        atomicAdd(scratch+numClasses+temp, 1);
    else
        atomicAdd(scratch+numClasses+numClasses+temp, 1);
    offset += numberThreads;
  }  
  __syncthreads ();

  if(threadIdx.x==0 && bid<numAttributes) {
    float entropy = 0;
    float entropy_l = 0;
    float entropy_r = 0;
    float proportion = 0;
    float proportion_l = 0;
    float proportion_r = 0;

    int numSamples_l = 0;
    for(int m = numClasses; m<2*numClasses; m++)
        numSamples_l += scratch[m];
    int numSamples_r = 0;
    for(int m = 2*numClasses; m<3*numClasses; m++)
        numSamples_r += scratch[m];


    for(int m=0; m<numClasses; m++){
        proportion = scratch[m]/numSamples;
        proportion_l = scratch[numClasses+m]/numSamples_l;
        proportion_r = scratch[2*numClasses+m]/numSamples_r;
        if(proportion!=0)
            entropy += (-1)*proportion*log2f(proportion);
        if(proportion_l!=0)
            entropy_l += (-1)*proportion_l*log2f(proportion_l);
        if(proportion_r!=0)
            entropy_r += (-1)*proportion_r*log2f(proportion_r);
     }
    g_odata[2*bid] = entropy - ((numSamples_l/numSamples)*entropy_l + (numSamples_r/numSamples)*entropy_r);
    g_odata[2*bid+1] = scratch[3*numClasses];
  }
  __syncthreads ();

}


void gpuHandler(float * samplesData, int * targetVals, int numSamples, int numAttrs, int numClasses, int * remAttributes, int & maxAttr, float & maxThreshold) {
    int numRequiredAttrs = 0;
    for(int i = 0; i<numAttrs; i++)
        if(remAttributes[i]==0)
            numRequiredAttrs++;

//    float * data_temp = (float *) malloc (numSamples * numRequiredAttrs * sizeof(float));
    int offset = 0;
    for(int i = 0; i< numAttrs; i++) {
        if(remAttributes[i]!=0)
            continue;
        for(int j = 0; j<numSamples; j++) {
            data_temp[offset] = samplesData[j*numAttrs+i];
            offset++;
        }
    }

//    float *d_output, *d_temp_attr;
//    int *d_target_values;
    /* allocate memory */
    //CUDA_CHECK_ERROR (cudaMalloc (&d_attr_values,numRequiredAttrs * numSamples  * sizeof (float)));
    //CUDA_CHECK_ERROR (cudaMalloc (&d_temp_attr, numRequiredAttrs * numSamples  * sizeof (float)));
    //CUDA_CHECK_ERROR (cudaMalloc (&d_target_values, numSamples  * sizeof (int)));
    //CUDA_CHECK_ERROR (cudaMalloc (&d_output, numRequiredAttrs * 2 * sizeof (float)));

//    CUDA_CHECK_ERROR (cudaMemcpy (d_attr_values, data_temp, numRequiredAttrs*numSamples * sizeof (float), cudaMemcpyHostToDevice));
//    CUDA_CHECK_ERROR (cudaMemcpy (d_temp_attr, data_temp, numRequiredAttrs*numSamples * sizeof (float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR (cudaMemcpy (d_target_values, targetVals, numSamples * sizeof (int), cudaMemcpyHostToDevice));
    
    /* GPU kernel */
    dim3 gb(MAX_BLOCKS, 1, 1);
    dim3 tb(MAX_THREADS, 1, 1);

    /* warm up */
//    kernel5 <<<gb, tb>>> (d_attr_values, d_temp_attr, d_target_values, d_output, numSamples, numClaDeviceSynchronize

    /* execute kernel */

    unsigned int attrPerBlock = MAX_BLOCKS;
    float * h_ans = (float *) malloc (2*numRequiredAttrs*sizeof(float));
    for(int iter = 0; iter < numRequiredAttrs; iter=iter+MAX_BLOCKS) {
        if(numRequiredAttrs - iter < MAX_BLOCKS)
            attrPerBlock = numRequiredAttrs - iter;
        // Copying max 64 attr data into CUDA
        CUDA_CHECK_ERROR (cudaMemcpy (d_attr_values, data_temp+iter*numSamples, attrPerBlock*numSamples * sizeof (float), cudaMemcpyHostToDevice));
        kernel5 <<<gb, tb>>> (d_attr_values, d_target_values, d_output, numSamples, numClasses, numRequiredAttrs);
        cudaDeviceSynchronize ();   

        /* copy result back from GPU */
        CUDA_CHECK_ERROR (cudaMemcpy (h_ans+iter*2, d_output, 2*attrPerBlock*sizeof (dtype), cudaMemcpyDeviceToHost));        
     
    }

#if 0
    kernel5 <<<gb, tb>>> (d_attr_values, d_target_values, d_output, numSamples, numClasses, numRDeviceSynchronize

    /* copy result back from GPU */
    float * h_ans = (float *) malloc (2*numRequiredAttrs*sizeof(float));
    CUDA_CHECK_ERROR (cudaMemcpy (h_ans, d_output, 2*numRequiredAttrs*sizeof (dtype), cudaMemcpyDeviceToHost));
#endif
    //printf("\nFinal Answer:- \n");
    //for(int k = 0; k<2*numRequiredAttrs; k++)
    //    printf(" %f ", h_ans[k]);

    int maxInd = -1;
    float max = 0;
    maxThreshold = -1;
    for(int j = 0; j<numRequiredAttrs; j++)
        if(h_ans[2*j]>max) {
            max = h_ans[2*j];
            maxInd = j;
            maxThreshold = h_ans[2*j+1];  
        }

    if(maxInd == -1) {
        maxAttr = -1;
        maxThreshold = -1;  
        free(h_ans);
        return;            
    }
    maxAttr = -1;
    int temp = -1;
    for(int j = 0; j < numAttrs; j++){
        if(remAttributes[j] == 0) {
            temp++;
            if(maxInd == temp) {
                maxAttr = j;
                free(h_ans);
               return; 
            }
        }
    }
}

//device
void alloc_cuda1D_float(int numSamples, int numAttrs)
{
    CUDA_CHECK_ERROR (cudaMalloc (&d_attr_values, MAX_BLOCKS * numSamples  * sizeof (float)));
    CUDA_CHECK_ERROR (cudaMalloc (&d_target_values, numSamples  * sizeof (int)));
    CUDA_CHECK_ERROR (cudaMalloc (&d_output, MAX_BLOCKS * 2 * sizeof (float)));
    data_temp = (float *) malloc (numSamples * numAttrs * sizeof(float));
}
