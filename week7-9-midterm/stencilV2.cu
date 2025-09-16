#include <stdio.h>
#define N 10
#define R 3
#define BLOCK_SIZE 4

__global__ void stencilV2(int *d_in,int *d_out,int n, int r){
    __shared__ int ds_in[BLOCK_SIZE + 2*R];
    int g_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int l_idx = threadIdx.x + r;
    if(g_idx < n){
        ds_in[l_idx] = d_in[g_idx];
        //left pad radius
        if(threadIdx.x < r && blockIdx.x ){
            ds_in[l_idx - r] = d_in[g_idx - r];
        }
        //right pad radius
        if(threadIdx.x >= blockDim.x - r){
            ds_in[l_idx + r] = d_in[g_idx + r];
        }
        __syncthreads();
        

        int t = 0;
        if(g_idx >= r && g_idx < n-r){
            for(int j = -r; j <= r;++j){
                t += ds_in[l_idx + j];
            }
        }
        d_out[g_idx] = t;

    }
}


int main(){
    int n = N,r = R,blockSize = BLOCK_SIZE;
    int block = (n + blockSize - 1)/blockSize;
    int *h_a,*h_b;
    h_a = (int*)malloc(n*sizeof(int));
    h_b = (int*)malloc(n*sizeof(int));
    for(int i = 0;i<n;++i)h_a[i] = i;
    int *d_a,*d_b;
    cudaMalloc((void**) &d_a,n*sizeof(int));
    cudaMalloc((void**) &d_b,n*sizeof(int));
    cudaMemcpy(d_a,h_a,n*sizeof(int),cudaMemcpyHostToDevice);

    stencilV2<<<block,blockSize>>>(d_a,d_b,n,r);
    cudaDeviceSynchronize();
    cudaMemcpy(h_b,d_b,n*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i = 0;i < n;++i){
        printf("OUT[%d] = %d\n",i,h_b[i]);
    }

    return 0;
}