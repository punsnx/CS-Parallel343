// 6610402230 Sirisuk Tharntham
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#define N 1e6
#define BLOCK_SIZE 2048
#define R 3

void stencilV0(int *h_in,int *h_out,int n,int r){
    if(n <= 2*r)return;
    for(int i = 0;i < n;++i){
        if(i >= r && i < n-r){
            int result = 0;
            for(int j = -r;j <= r;++j){
                result += h_in[i + j];
            }
            h_out[i] = result;
        }
    }
}

__global__ void stencilV1(int *d_in,int *d_out,int n,int r){
    if(n <= 2*r)return;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int result = 0;
    if(idx >= r && idx < n-r){
        for(int j = -r;j <= r;++j)
            result += d_in[idx + j];
        
        d_out[idx] = result;
    }else{
        d_out[idx] = 0;
    }
}



__global__ void stencilV2(int *d_in,int *d_out,int n,int r){
    if(n <= 2*r)return;
    __shared__ int cache[BLOCK_SIZE + 2*R];
    int g_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int l_idx = threadIdx.x+r;
    if(g_idx < n){
        cache[l_idx] = d_in[g_idx];


        if(threadIdx.x < r && blockIdx.x > 0){
            cache[l_idx - r] = d_in[g_idx - r];
        }
        if(threadIdx.x >= blockDim.x - r){
            cache[l_idx + r] = d_in[g_idx + r];
        }

        __syncthreads();

        if(g_idx >= r && g_idx < n-r){
            int result = 0;
            for(int j = -r;j <= r;++j){
                result += cache[l_idx + j];
            }
            d_out[g_idx] = result;
        }else{
            d_out[g_idx] = 0;
        }


        
    }





}

double processTime(clock_t start,clock_t stop){
    return ((double)(stop - start))/CLOCKS_PER_SEC;
}

float processGpuTime(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds / 1000.0f; // Convert to seconds
}

int main(){

    int n = N;
    int r = R;
    int block_size = BLOCK_SIZE;

    // int block = ( n % block_size ? n/block_size + 1 : n % block_size);
    int block = (n + block_size - 1) / block_size;

    clock_t start,stop;

    int *h_in,*h_out;
    h_in = (int *)malloc(n * sizeof(int));
    h_out = (int *)malloc(n * sizeof(int));

    for(int i = 0;i < n;++i){
        h_in[i] = i;
    }

    start = clock();
    stencilV0(h_in,h_out,n,r);
    stop = clock();

    double timeV0 = processTime(start,stop);

    // for(int i = 0;i < n;++i)
        // printf("%d %d\n",i,h_out[i]);


    printf("TIME V0: %.17f\n",timeV0);

    int *d_in,*d_out;
    cudaMalloc((void **) &d_in, n * sizeof(int));
    cudaMalloc((void **) &d_out,n * sizeof(int));
    cudaMemcpy(d_in,h_in,n*sizeof(int),cudaMemcpyHostToDevice);
    //V1
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);cudaEventCreate(&gpu_stop);

    cudaEventRecord(gpu_start);
    stencilV1<<<block,block_size>>>(d_in,d_out,n,r);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    cudaDeviceSynchronize();
    
    double timeV1 = processGpuTime(gpu_start,gpu_stop);
    cudaMemcpy(h_out,d_out,n*sizeof(int),cudaMemcpyDeviceToHost);

    // for(int i = 0;i < n;++i)
        // printf("%d %d\n",i,h_out[i]);
    printf("TIME V1: %.17f\n",timeV1);

    //V2

    cudaEvent_t gpu_start2, gpu_stop2;
    cudaEventCreate(&gpu_start2);
    cudaEventCreate(&gpu_stop2);

    cudaEventRecord(gpu_start2);
    stencilV2<<<block,block_size>>>(d_in,d_out,n,r);
    cudaEventRecord(gpu_stop2);
    cudaEventSynchronize(gpu_stop2);
    cudaDeviceSynchronize();
    
    double timeV2 = processGpuTime(gpu_start2,gpu_stop2);
    cudaMemcpy(h_out,d_out,n*sizeof(int),cudaMemcpyDeviceToHost);

    // for(int i = 0;i < n;++i)
        // printf("%d %d\n",i,h_out[i]);
    printf("TIME V2 (Cache): %.17f\n",timeV2);


    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}