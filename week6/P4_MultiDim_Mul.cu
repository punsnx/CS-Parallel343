//6610402230 Sirisuk Tharntham
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#define N 200
#define TILE_WIDTH 16 //as block_size

//CPU version
void matrixMul2DV0(double *h1_in,double *h2_in,double *h_out, int n){
    for(int r = 0;r < n; ++r){
        for(int c = 0; c < n;++c){
            double product = 0;
            for(int k = 0; k < n;++k){
                    product += h1_in[r * n + k] * h2_in[k * n + c];
            }
            h_out[r * n + c] = product;

        }
    }

}

//CUDA version - global memory
__global__ void matrixMul2DV1(double *d1_in,double *d2_in,double *d_out, int n){
    int r = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.x * blockIdx.x + threadIdx.x;

    if(r < n && c < n){
        double product = 0;
        for(int k = 0;k < n;++k){
            product += d1_in[r * n + k] * d2_in[k * n + c];
        }
        d_out[r * n + c] = product;
    }



}

//CUDA version - share memory
__global__ void matrixMul2DV2(double *d1_in,double *d2_in,double *d_out, int n){
    __shared__ double ds_M[TILE_WIDTH * TILE_WIDTH];
    __shared__ double ds_N[TILE_WIDTH * TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int r = blockDim.y * blockIdx.y + ty; 
    int c = blockDim.x * blockIdx.x + tx;

    double product = 0;//product value)
    for(int p = 0;p < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++p){
        int g_c = p * TILE_WIDTH + tx;
        int g_r = p * TILE_WIDTH + ty;
        ds_M[ty * TILE_WIDTH + tx] = ( (r < n && g_c < n) ? d1_in[r * n + g_c] : 0);
        ds_N[ty * TILE_WIDTH + tx] = ( (g_r < n && c < n) ? d2_in[g_r * n + c] : 0);
        // value of the index that out of range setted to large value ensure 
        // no incorrect memory access in the threads

        __syncthreads();

        for(int i = 0;i < TILE_WIDTH;++i){
            if(p * TILE_WIDTH + i < n){
                product += ds_M[ty * TILE_WIDTH + i] * ds_N[i * TILE_WIDTH + tx];
            }
        }
        __syncthreads();
    }
    if(r < n && c < n)
        d_out[r * n + c] = product;

}


int isMetrixEqual(double *A, double *B,int n){
    for(int r = 0;r<n;++r){
        for(int c = 0;c<n;++c){
            if(A[r * n + c] != B[r * n + c])return 0;
        }
    }

    return 1;
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
    //matrix multiplication with multi-dim threads excecution
    int n = N;
    double *h1_in = (double *)malloc(n*n*sizeof(double));
    double *h2_in = (double *)malloc(n*n*sizeof(double));
    double *h_out = (double *)malloc(n*n*sizeof(double));

    // Assign value
    for(int r = 0;r<n;++r){
        for(int c = 0;c<n;++c){
            h1_in[r * n + c] = r * n + c;
            h2_in[r * n + c] = r * n + c;
        }
    }
    // for(int r = 0;r<n;++r){
    //     for(int c = 0;c<n;++c){
    //         printf("%.2f ",h1_in[r * n + c]);
    //     }
    //     printf("\n");
    // }
    // for(int r = 0;r<n;++r){
    //     for(int c = 0;c<n;++c){
    //         printf("%.2f ",h2_in[r * n + c]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // V0 CPU Version
    clock_t start = clock();
    matrixMul2DV0(h1_in,h2_in,h_out,n);
    clock_t end = clock();
    double timeV0 = processTime(start,end);


    // for(int r = 0;r<n;++r){
    //     for(int c = 0;c<n;++c){
    //         printf("%.2f ",h_out[r * n + c]);
    //     }
    //     printf("\n");
    // }

    double *d1_in,*d2_in,*d_out;
    cudaMalloc((void**) &d1_in,n*n*sizeof(double));
    cudaMalloc((void**) &d2_in,n*n*sizeof(double));
    cudaMalloc((void**) &d_out,n*n*sizeof(double));

    cudaMemcpy(d1_in,h1_in,n*n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d2_in,h2_in,n*n*sizeof(double),cudaMemcpyHostToDevice);

    // initialize kernel value
    int block_size = TILE_WIDTH; // limit to hardware max thread/block 32*32 = 1024
    int block = (n + block_size - 1) / block_size;
    
    dim3 dimGrid(block,block,1);
    dim3 dimBlock(block_size,block_size,1);

    // V1 Global memory

    double *h2_out = (double *)malloc(n*n*sizeof(double));

    cudaEvent_t startV1,endV1; cudaEventCreate(&startV1);cudaEventCreate(&endV1);
    cudaEventRecord(startV1);
    matrixMul2DV1<<<dimGrid,dimBlock>>>(d1_in,d2_in,d_out,n);
    cudaDeviceSynchronize();
    cudaEventRecord(endV1);
    cudaMemcpy(h2_out,d_out,n*n*sizeof(double),cudaMemcpyDeviceToHost);
    float timeV1 = processGpuTime(startV1,endV1);

    // for(int r = 0;r<n;++r){
    //     for(int c = 0;c<n;++c){
    //         printf("%.2f ",h2_out[r * n + c]);
    //     }
    //     printf("\n");
    // }
    
    //CUDA V2 share memory
    
    double *h3_out = (double *)malloc(n*n*sizeof(double));
    
    cudaEvent_t startV2,endV2; cudaEventCreate(&startV2);cudaEventCreate(&endV2);
    cudaEventRecord(startV2);
    matrixMul2DV2<<<dimGrid,dimBlock>>>(d1_in,d2_in,d_out,n);
    cudaDeviceSynchronize();
    cudaEventRecord(endV2);
    cudaMemcpy(h3_out,d_out,n*n*sizeof(double),cudaMemcpyDeviceToHost);
    float timeV2 = processGpuTime(startV2,endV2);
    
    
    for(int r = 0;r<n;++r){
        for(int c = 0;c<n;++c){
            printf("%.2f",h3_out[r * n + c]);
        }
        printf("\n");
    }
    
    printf("V1 Matrix is %s\n",( isMetrixEqual(h_out,h2_out,n) ? "Equal" : "Not Equal"));
    printf("V2 Matrix is %s\n",( isMetrixEqual(h_out,h3_out,n) ? "Equal" : "Not Equal"));
    printf("Time V0 : %.9f\n", timeV0);
    printf("Time V1 : %.9f\n", timeV1);
    printf("Time V2 : %.9f\n", timeV2);
    
    return 0;

}

