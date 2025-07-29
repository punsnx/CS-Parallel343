//6610402230 Sirisuk Tharntham
#include <stdio.h>

__global__ void vectorAdd(int *d_a, int *d_b, int *d_c,int n){

    // dynamicly address index by device block
    int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    if(idx < n){
        d_c[idx] = d_a[idx] + d_b[idx];
        //look up which thread is running? 
        printf("Block: %u, thread: %u - ARR[%d]: %d\n", blockIdx.x, threadIdx.x,idx,d_c[idx]);

    }

}

int main(){

    int n = 1000;
    int block_size = 512;
    int block = ( n % block_size ? n/block_size + 1 : n % block_size);

    //part 0 allocate host memory
    int *h_a,*h_b,*h_c;
    h_a = (int *)malloc(n*sizeof(int));
    h_b = (int *)malloc(n*sizeof(int));
    h_c = (int *)malloc(n*sizeof(int));

    //part 1 sequential addition
    for(int i = 0; i < n; i++){
        h_a[i] = i;
        h_b[i] = i;
        h_c[i] = h_a[i] + h_b[i];

        printf("ARR_C[%d] : %d\n",i,h_c[i]);

    }
    //confirm parallel memory works not from sequential
    free(h_c);
    h_c = (int *)malloc(n*sizeof(int));

    printf("=================\n");
    //part 2 allocation device memory
    int *d_a,*d_b,*d_c;
    cudaMalloc((void**) &d_a,n*sizeof(int));
    cudaMalloc((void**) &d_b,n*sizeof(int));
    cudaMalloc((void**) &d_c,n*sizeof(int));

    cudaMemcpy(d_a,h_a,n*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,n*sizeof(int),cudaMemcpyHostToDevice);

    //part 4 parallel addition
    vectorAdd<<<block,block_size>>>(d_a,d_b,d_c,n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c,d_c,n*sizeof(int),cudaMemcpyDeviceToHost);
    printf("=================\n");
    for(int i = 0;i < n;++i){
        printf("CUDA - ARR_C[%d] : %d\n",i,h_c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}