#include <stdio.h>

__global__ void stencilV1(int *d_in,int *d_out,int n, int r){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= r && idx < n-r){
        int t = 0;
        for(int i = -r; i <= r;++i)
            t += d_in[idx + i];
        d_out[idx] = t;
    }
}


int main(){
    int n = 10,r = 3,blockSize = 8;
    int block = (n + blockSize - 1)/blockSize;
    int *h_a,*h_b;
    h_a = (int*)malloc(n*sizeof(int));
    h_b = (int*)malloc(n*sizeof(int));
    for(int i = 0;i<n;++i)h_a[i] = i;
    int *d_a,*d_b;
    cudaMalloc((void**) &d_a,n*sizeof(int));
    cudaMalloc((void**) &d_b,n*sizeof(int));
    cudaMemcpy(d_a,h_a,n*sizeof(int),cudaMemcpyHostToDevice);

    stencilV1<<<block,blockSize>>>(d_a,d_b,n,r);
    cudaDeviceSynchronize();
    cudaMemcpy(h_b,d_b,n*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i = 0;i < n;++i){
        printf("OUT[%d] = %d\n",i,h_b[i]);
    }

    return 0;
}