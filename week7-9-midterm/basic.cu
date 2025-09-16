#include <stdio.h>


__global__ void kernelFunc(){
    printf("Hello block %u %thread %u\n", blockIdx.x,threadIdx.x);
}


int main(){
    //block,blockSize
    kernelFunc<<<9,4>>>();
    cudaDeviceSynchronize();
    //see that print blocks and threads print indepentdent in SIMD
    //not sequential order 

    return 0;
}