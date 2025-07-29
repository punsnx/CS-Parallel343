#include <stdio.h>
using namespace std;

void vectorAdd2DV0(int *h1_in,int *h2_in,int *h_out,int n){
    for(int r =0;r< n;++r){
        for(int c = 0;c<n;++c){
            h_out[r * n + c] = h1_in[r * n + c] + h2_in[r * n + c];
        }
    }
    
}

__global__ void  vectorAdd2D(int *d1_in,int *d2_in,int *d_out,int n){
    int r = blockDim.x * blockIdx.x + threadIdx.x;
    int c = blockDim.y * blockIdx.y + threadIdx.y;

    if(r < n && c < n){
            d_out[r * n + c] = d1_in[r * n + c] + d2_in[r * n + c];
    }
}

int isMetrixEqual(int *N, int *M,int n){
    for(int r = 0;r<n;++r){
        for(int c = 0;c<n;++c){
            if(N[r * n + c] != M[r * n + c])return 0;
        }
    }

    return 1;
}



int main(){
    int n = 20;
    int *h1_in = (int *)malloc(n*n*sizeof(int));
    int *h2_in = (int *)malloc(n*n*sizeof(int));
    int *h_out = (int *)malloc(n*n*sizeof(int));

    for(int r = 0;r<n;++r){
        for(int c = 0;c<n;++c){
            h1_in[r * n + c] = r * n + c;
            h2_in[r * n +c ] = r * n + c;
        }
    }

    vectorAdd2DV0(h1_in,h2_in,h_out,n);


    // for(int r = 0;r<n;++r){
    //     for(int c = 0;c<n;++c){
    //         printf("%d ",h_out[r * n + c]);
    //     }
    //     printf("\n");
    // }

    int *d1_in,*d2_in,*d_out;
    cudaMalloc((void**) &d1_in,n*n*sizeof(int));
    cudaMalloc((void**) &d2_in,n*n*sizeof(int));
    cudaMalloc((void**) &d_out,n*n*sizeof(int));

    cudaMemcpy(d1_in,h1_in,n*n*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d2_in,h2_in,n*n*sizeof(int),cudaMemcpyHostToDevice);

    // call kernel
    int block_size = 32; // limit to hardware max thread/block 32*32 = 1024
    int block = (n + block_size - 1) / block_size;

    dim3 dimGrid(block,block,1);
    dim3 dimBlock(block_size,block_size,1);

    int *h2_out = (int *)malloc(n*n*sizeof(int));

    vectorAdd2D<<<dimGrid,dimBlock>>>(d1_in,d2_in,d_out,n);
    cudaDeviceSynchronize();
    cudaMemcpy(h2_out,d_out,n*n*sizeof(int),cudaMemcpyDeviceToHost);




    for(int r = 0;r<n;++r){
        for(int c = 0;c<n;++c){
            printf("%d ",h2_out[r * n + c]);
        }
        printf("\n");
    }

    printf("Matrix is %s",( isMetrixEqual(h_out,h2_out,n) ? "Equal" : "Not Equal"));

    //matrix multiplication


    











    return 0;
}