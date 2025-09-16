#include <iostream>
using namespace std;
#define N 10
#define BLOCK_SIZE 3


// __global__ void reduction(int *d_in,int size){
//     __shared__ int ds_in[2 * BLOCK_SIZE];
//     int t = threadIdx.x; 
//     int idx = 2 * blockDim.x * blockIdx.x + t;
//     ds_in[t] = ( idx < size ? d_in[idx] : 0);
//     ds_in[BLOCK_SIZE + t] = ( BLOCK_SIZE + idx < size ? d_in[BLOCK_SIZE + idx] : 0);
//     __syncthreads();
//     // printf("Block (%d), Thread (%d): t=%d, DATA %d %d\n", blockIdx.x, threadIdx.x, t, ds_in[t], ds_in[BLOCK_SIZE + t]);

//     for(int s = 1; s < 2 * BLOCK_SIZE;s*=2){
//         if(t % s == 0){
//             ds_in[2 * t] += ds_in[2 * t + s];
//         }
//         __syncthreads();
//     }
//     // for(int s = BLOCK_SIZE;s > 0;s /= 2){
//     //     if(s % 2 && t == BLOCK_SIZE - 1){
//     //         ds_in[0] 
//     //     }
//     //     if(t < s){
//     //         ds_in[t] += ds_in[t+s];

//     //     }
//     //     __syncthreads();
//     // }
//     // printf("--------------\n");
//     // printf("Block (%d), Thread (%d): t=%d, DATA %d %d\n", blockIdx.x, threadIdx.x, t, ds_in[t], ds_in[BLOCK_SIZE + t]);
//     if(t == 0 && idx < size)d_in[idx] = ds_in[t];
// }

__global__ void reduction(int *d_in,int size){
    __shared__ int ds_in[BLOCK_SIZE];
    int t = threadIdx.x; 
    int idx = blockDim.x * blockIdx.x + t;
    ds_in[t] = ( idx < size ? d_in[idx] : 0);
    __syncthreads();
    printf("Block (%d), Thread (%d): t=%d, DATA %d\n", blockIdx.x, threadIdx.x, t, ds_in[t]);

    for(int s = 1; s < BLOCK_SIZE;s*=2){
        if(t % (2*s) == 0){
            ds_in[t] += ds_in[t+s];
        }
        __syncthreads();
    }
    printf("--------------\n");
    printf("Block (%d), Thread (%d): t=%d, DATA %d\n", blockIdx.x, threadIdx.x, t, ds_in[t]);
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

    reduction<<<dimGrid,dimBlock>>>(d_a,n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_a,d_a,n * sizeof(int),cudaMemcpyDeviceToHost);
    int sum = 0;
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