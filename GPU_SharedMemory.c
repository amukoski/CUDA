#include <stdio.h>

#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__

struct GpuTimer
{
      cudaEvent_t start;
      cudaEvent_t stop;
 
      GpuTimer()
      {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
      }
 
      ~GpuTimer()
      {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
      }
 
      void Start()
      {
            cudaEventRecord(start, 0);
      }
 
      void Stop()
      {
            cudaEventRecord(stop, 0);
      }
 
      float Elapsed()
      {
            float elapsed;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            return elapsed;
      }
};

#endif
//Shared
__global__ void stringMatchingGPU(char *data, unsigned int dataLen, char *target, unsigned int targetLen, unsigned int *matchPositions, unsigned int *numOfMatches){
    __shared__ char shmemData[1024*2];
    __shared__ char shmemTarget[1024];
    
    unsigned int index = blockDim.x*blockIdx.x + threadIdx.x;
    
    shmemData[threadIdx.x] = data[index];
    shmemData[threadIdx.x+targetLen] = data[(index+targetLen) % dataLen];
    shmemTarget[threadIdx.x] = target[threadIdx.x];
    __syncthreads();
    
    if(index+targetLen > dataLen) return;
    if(shmemData[threadIdx.x] == shmemTarget[0]){

        int flag = 1;
        for(unsigned int i=1; i<targetLen; i++){
            if(shmemData[threadIdx.x+i] != shmemTarget[i]){
                flag = 0; break;
            }
        }
        
        if(flag == 1){
            matchPositions[atomicAdd(numOfMatches,1)] = index;
        }
    }
    
}

int main(int argc,char **argv)
{   
    double timerElapsed = 0.0;
    for(int N=0;N<1000;N++){
		GpuTimer timer;
		
		// declare and allocate host memory
		char *h_data = (char *) malloc(1*1024*1024*sizeof(char));    // 1MB char array
        char *h_target = (char *) malloc(1*1024*1024*sizeof(char));  // 1MB char array
        strcpy(h_data, "Lorem ipsum adore itom Lorem ipsum");
		strcpy(h_target, "ipsum");
		
		unsigned int h_dataLen = 0;
        while(h_data[h_dataLen++] != '\0');
        --h_dataLen;
        
        unsigned int h_targetLen = 0;
        while(h_target[h_targetLen++] != '\0');
        --h_targetLen;
        
		unsigned int *h_matchPositions = (unsigned int *) malloc(h_dataLen*sizeof(unsigned int));
        unsigned int *h_numOfMatches = (unsigned int *) malloc(sizeof(unsigned int));
		
		*h_numOfMatches = 0;
		
		// declare, allocate, and zero out GPU memory
		char *d_data;
		char *d_target;
		unsigned int  *d_matchPositions;
		unsigned int  *d_numOfMatches;
		
		cudaMalloc((void **) &d_data, h_dataLen*sizeof(char));
		cudaMalloc((void **) &d_target, h_targetLen*sizeof(char));    
		cudaMalloc((void **) &d_matchPositions, h_dataLen * sizeof(unsigned int));
		cudaMalloc((void **) &d_numOfMatches, sizeof(unsigned int));    
		
		cudaMemcpy(d_data, h_data, h_dataLen*sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(d_target, h_target, h_targetLen*sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(d_numOfMatches, h_numOfMatches, sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemset((void *) d_matchPositions, 0, h_dataLen * sizeof(unsigned int)); 

		// launch the kernel - comment out one of these
		timer.Start();
			stringMatchingGPU<<<h_dataLen/h_targetLen,h_targetLen>>>(d_data, h_dataLen, d_target, h_targetLen, d_matchPositions, d_numOfMatches);
		timer.Stop();
		
		// copy back the array of sums from GPU and print
		cudaMemcpy(h_matchPositions, d_matchPositions, h_dataLen * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_numOfMatches, d_numOfMatches, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		
		//printf("Number Of Matches = %d\n", *h_numOfMatches);
		/*
		for(unsigned int i=0; i<*h_numOfMatches; i++)
			printf("%d\n", h_matchPositions[i]);
		*/
		
		timerElapsed += timer.Elapsed();
		// free GPU memory allocation and exit
		cudaFree(d_data);
		cudaFree(d_target);
		cudaFree(d_matchPositions);
		cudaFree(d_numOfMatches);
		
		free(h_data);
        free(h_target);
        free(h_matchPositions);
        free(h_numOfMatches);
    }
    
    printf("Time elapsed = %g ms\n", timerElapsed/1000.0);

    return 0;
}