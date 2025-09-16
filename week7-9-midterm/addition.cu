#include <stdio.h>

__global__ void addition(int *d_a,int *d_b,int *d_c,int n){
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n){
        d_c[idx] = d_a[idx] + d_b[idx];
    }
}
int main(){
    int n = 10,blockSize = 8;
    int block = (n + blockSize - 1)/blockSize;
    int *h_a,*h_b,*h_c;
    h_a = (int*)malloc(n*sizeof(int));
    h_b = (int*)malloc(n*sizeof(int));
    h_c = (int*)malloc(n*sizeof(int));
    for(int i = 0;i<n;++i)h_a[i] = h_b[i] = i;
    int *d_a,*d_b,*d_c;
    cudaMalloc((void**) &d_a,n*sizeof(int));
    cudaMalloc((void**) &d_b,n*sizeof(int));
    cudaMalloc((void**) &d_c,n*sizeof(int));
    cudaMemcpy(d_a,h_a,n*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,n*sizeof(int),cudaMemcpyHostToDevice);

    addition<<<block,blockSize>>>(d_a,d_b,d_c,n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c,d_c,n*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i = 0;i<n;++i){
        printf("%d + %d = %d\n",h_a[i],h_b[i],h_c[i]);
    }












    return 0;
}