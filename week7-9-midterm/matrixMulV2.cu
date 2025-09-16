#include <stdio.h>
#define N 3
#define TILE_WIDTH 2
__global__ void matrixMulV2(int *d_a,int *d_b,int *d_c,int n){
    __shared__ int ds_M[TILE_WIDTH * TILE_WIDTH];
    __shared__ int ds_N[TILE_WIDTH * TILE_WIDTH];

    int r = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int t = 0;
    for(int p = 0;p < ( n + TILE_WIDTH - 1)/TILE_WIDTH;++p){
        int pr = p * TILE_WIDTH + ty;
        int pc = p * TILE_WIDTH + tx;
        ds_M[ty * TILE_WIDTH + tx] = (r < n && pc < n ? d_a[r * n + pc] : 0);
        ds_N[ty * TILE_WIDTH + tx] = (pr < n && c < n ? d_b[pr * n + c] : 0);
        __syncthreads();
        for(int i = 0;i < TILE_WIDTH;++i){
            if(p * TILE_WIDTH + i < n){
                t += ds_M[ty * TILE_WIDTH + i] * ds_N[i * TILE_WIDTH + tx];
            }
        }
        __syncthreads();
    }
    if(r < n && c < n){
        d_c[r * n + c] = t;
    }






}
int main(){

    int n = N,blockSize = TILE_WIDTH;
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

    matrixMulV2<<<dimGrid,dimBlock>>>(d_a,d_b,d_c,n);
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