#include <stdio.h>


__global__ void matrixAdd(int *d_a,int *d_b,int *d_c,int n){
    int r = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    if(r < n && c < n){
        int i = r*n+c;
        d_c[i] = d_a[i] + d_b[i];
    }
}
int main(){
    int n = 5,blockSize = 4;
    int block = (n + blockSize - 1)/blockSize;
    dim3 dimGrid(block,block,1);
    dim3 dimBlock(blockSize,blockSize,1);
    int *h_a,*h_b,*h_c;
    h_a = (int*)malloc(n*n*sizeof(int));
    h_b = (int*)malloc(n*n*sizeof(int));
    h_c = (int*)malloc(n*n*sizeof(int));
    for(int i = 0;i<n;++i)
        for(int j = 0;j<n;++j)h_a[i * n + j] = h_b[i * n + j] = i * n + j;
    int *d_a,*d_b,*d_c;
    cudaMalloc((void**) &d_a,n*n*sizeof(int));
    cudaMalloc((void**) &d_b,n*n*sizeof(int));
    cudaMalloc((void**) &d_c,n*n*sizeof(int));
    cudaMemcpy(d_a,h_a,n*n*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,n*n*sizeof(int),cudaMemcpyHostToDevice);

    matrixAdd<<<dimGrid,dimBlock>>>(d_a,d_b,d_c,n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c,d_c,n*n*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i = 0;i<n;++i){
        for(int j = 0;j<n;++j)
            printf("%d ",h_a[i * n + j]);
        printf("\n");
    }
    printf("\n");
    for(int i = 0;i<n;++i){
        for(int j = 0;j<n;++j)
            printf("%d ",h_b[i * n + j]);
        printf("\n");
    }
    printf("\n");

    for(int i = 0;i<n;++i){
        for(int j = 0;j<n;++j)
            printf("%d ",h_c[i * n + j]);
        printf("\n");
        
    }

    return 0;
}