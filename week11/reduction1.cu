// 6610402230 Sirisuk Tharntham
#include <iostream>
using namespace std;
#define N 1024 * 1024
#define BLOCK_SIZE 16

__device__ void debug(int *mem,int idx){
    int b = blockIdx.x;
    int t = threadIdx.x;
    printf("===============\n");
    printf("Block (%d), Thread (%d) DATA %d\n", b, t, mem[idx]);
}
// global memory reduction
__global__ void reductionV1(int *d_in,int size){
    int t = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size)return;
    // debug(d_in,idx);
    for(int s = 1;s < BLOCK_SIZE;s*=2){
        if(t % (2*s) == 0 && idx < size){
            if(idx + s < blockDim.x * (blockIdx.x + 1))//bound at same block
            d_in[idx] += d_in[idx+s];
        }
        __syncthreads();
    }
    // debug(d_in,idx);

}
//Interleaved addressing
__global__ void reductionV2(int *d_in,int size){
    __shared__ int ds_in[BLOCK_SIZE];
    int t = threadIdx.x; 
    int idx = blockDim.x * blockIdx.x + t;
    ds_in[t] = ( idx < size ? d_in[idx] : 0);
    __syncthreads();
    // debug(ds_in,t);

    for(int s = 1; s < BLOCK_SIZE;s*=2){
        if(t % (2*s) == 0){
            ds_in[t] += ds_in[t+s];
        }
        __syncthreads();
    }

    // debug(ds_in,t);
    

    if(t == 0 && idx < size)d_in[idx] = ds_in[t];
}
//Interleaved addressing 
__global__ void reductionV3(int *d_in,int size){
    __shared__ int ds_in[BLOCK_SIZE];
    int t = threadIdx.x; 
    int idx = blockDim.x * blockIdx.x + t;
    ds_in[t] = ( idx < size ? d_in[idx] : 0);
    __syncthreads();
    // debug(ds_in,t);

    for(int s = 1; s < BLOCK_SIZE;s*=2){
        int i = 2 * s * t;
        if(i < BLOCK_SIZE){
            ds_in[i] += ds_in[i+s];
        }
        __syncthreads();
    }

    // debug(ds_in,t);
    

    if(t == 0 && idx < size)d_in[idx] = ds_in[t];
} 
//sequential addressing - work with N of size 2^k complete binary tree
// problem is missing last odd value
//may be bugs
__global__ void reductionV4(int *d_in,int size){
    __shared__ int ds_in[BLOCK_SIZE];
    int t = threadIdx.x; 
    int idx = blockDim.x * blockIdx.x + t;
    ds_in[t] = ( idx < size ? d_in[idx] : 0);
    __syncthreads();
    // debug(ds_in,t);

    for(int s = BLOCK_SIZE; s >0; s >>= 1){
        if(t < s){
            ds_in[t] += ds_in[t+s];
        }
        __syncthreads();
    }

    // debug(ds_in,t);
    

    if(t == 0 && idx < size)d_in[idx] = ds_in[t];
}

int main(){
    int n = N;
    int block_size = BLOCK_SIZE;
    int block = (n + block_size - 1) / block_size;
    dim3 dimGrid(block,1,1);
    dim3 dimBlock(block_size,1,1);

    int *h_a = new int[n];
    for(int i = 0;i < n;++i)h_a[i] = i;
    int *d_a;
    cudaMalloc(&d_a,n * sizeof(int));
    cudaMemcpy((void *)d_a, h_a,n * sizeof(int),cudaMemcpyHostToDevice);

    reductionV4<<<dimGrid,dimBlock>>>(d_a,n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_a,d_a,n * sizeof(int),cudaMemcpyDeviceToHost);
    long sum = 0;
    for(int i = 0;i < block;++i){
        int idx = i * block_size;
        if(idx < n){
            cout << "IDX " << idx << "DATA" << h_a[idx] << endl;
            sum += h_a[idx];

        }
    }
    cout << "SUM " << sum << endl;



    return 0;
}