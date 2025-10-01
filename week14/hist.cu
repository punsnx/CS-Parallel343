// 6610402230 Sirisuk Tharntham
#include <iostream>
#define N 1000000
#define NUM_BINS 128
#define MAX_COUNT 127
#define BLOCK_SIZE 16
using namespace std;

__global__ void hist_kernel(u_int16_t *d_in,long num_in,u_int32_t *d_bin,long num_bins){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while(i < num_in){ 
        if(d_in[i] < num_bins){ //limit from 0 to num_bins - 1
            u_int32_t old = atomicAdd(&d_bin[d_in[i]],1);
            //use clean below with atomic sub or use hist_clean instead to verify MAX_COUNT
            //BUT use atomic is cost expensive
            // if(old >= MAX_COUNT)atomicSub(&d_bin[d_in[i]],1); // old return by atomic r-m-w prevent race 
        }
        i += stride;
    }
}

__global__ void hist_clean(u_int32_t *d_bin,long num_bins){
    long i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < num_bins)
        d_bin[i] = (d_bin[i] > MAX_COUNT ? MAX_COUNT : d_bin[i]);

}

__global__ void hist_kernelV2(u_int16_t *d_in,long num_in,u_int32_t *d_bin,long num_bins){
    __shared__ u_int32_t ds_private_bin[NUM_BINS];
    int t = threadIdx.x;
    int part = (num_bins + blockDim.x - 1) / blockDim.x;
    for(int p = 0;p < part;++p){
        int idx = p * blockDim.x + t;
        if(idx < num_bins) ds_private_bin[idx] = 0;
    }
    __syncthreads();
    int stride = blockDim.x * gridDim.x;
    int i = blockDim.x * blockIdx.x + t;
    while(i < num_in){
        if(d_in[i] < num_bins)
            atomicAdd(&ds_private_bin[d_in[i]],1);
        i += stride;
    }
    __syncthreads();

    for(int p = 0;p < part;++p){
        int idx = p * blockDim.x + t;
        if(idx < num_bins) atomicAdd(&d_bin[idx],ds_private_bin[idx]);
    }
}



void histrogram(u_int16_t *in,long num_in, u_int32_t *bin, long num_bins){

    int block_size = BLOCK_SIZE;
    int block = (num_in + block_size - 1) / block_size;
    dim3 dimGrid(block,1,1);
    dim3 dimBlock(block_size,1,1);

    u_int32_t *d_bin;
    u_int16_t *d_in;
    cudaMalloc((void**) &d_bin,num_bins *sizeof(u_int32_t));
    cudaMalloc((void**) &d_in,num_in *sizeof(u_int16_t));
    cudaMemcpy(d_bin,bin,num_bins * sizeof(u_int32_t),cudaMemcpyHostToDevice);
    cudaMemcpy(d_in,in,num_in * sizeof(u_int16_t),cudaMemcpyHostToDevice);

    hist_kernelV2<<<dimGrid,dimBlock>>>(d_in,num_in,d_bin,num_bins);

    block_size = BLOCK_SIZE;
    block = (num_bins + block_size - 1) / block_size;
    dimGrid = dim3(block,1,1);
    dimBlock = dim3(block_size,1,1);

    //For practice 7.2 not saturate, so leave comment hist_clean
    // hist_clean<<<dimGrid,dimBlock>>>(d_bin,num_bins);

    cudaMemcpy(bin,d_bin,num_bins * sizeof(u_int32_t),cudaMemcpyDeviceToHost);
    // cudaMemcpy(in,d_in,num_in * sizeof(u_int16_t),cudaMemcpyDeviceToHost); //Not nessessary

}


int main(){
    long n = N;
    long bin = NUM_BINS;

    u_int32_t *h_bin = new u_int32_t[bin];
    u_int16_t *h_in = new u_int16_t[n];

    for(int i = 0;i < bin;++i){
        h_bin[i] = 0;
    }
    
    //Practice 7.1 => SET N any BIN 4096
    //Practice 7.2 => SET N any BIN 128
    for(int i = 0;i < n;++i){
        h_in[i] = i%bin;
        // h_in[i] = 4000;
    }

    histrogram(h_in,n,h_bin,bin);

    //practice 7.1
    // for(int i = 0;i < bin;++i){
    //     cout << "BIN " << i << " = " << h_bin[i] << endl;

    // }

    //practice 7.2
    //try add (char) h_bin[i] and SET N 128*65 = 8320 will see 'A'
    for(int i = 0;i < bin;++i){
        cout << "BIN " << i << " = " << h_bin[i] << endl;

    }
    return 0;
}