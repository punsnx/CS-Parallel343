// 6610402230 Sirisuk Tharntham
#include <iostream>
using namespace std;
#define N 32
#define BLOCK_SIZE 8

enum OPERATION {
    SUM = 0,
    PRODUCT = 1,
    MAX = 2,
    MIN = 3
};

__device__ __host__ int getIdentity(OPERATION &o) {
    switch(o) {
        case SUM: return 0;        
        case PRODUCT: return 1;  
        case MAX: return INT_MIN;   
        case MIN: return INT_MAX;
        default: return 0;
    }
}

__device__ __host__ void process(int *o1,int *o2,OPERATION o){
    if(o == SUM)  *o1 += *o2;
    else if(o == PRODUCT) *o1 *= *o2;
    else if (o == MIN) *o1 = (*o1 < *o2 ? *o1 : *o2);
    else if (o == MAX) *o1 = (*o1 > *o2 ? *o1 : *o2);
    else return;
}
// Hillis and Steele Algorithm (Inclusive Scan)
__global__ void scanV1(int *d_in,int size,OPERATION o){
    __shared__ int ds_in[BLOCK_SIZE];
    int t = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size)return;
    ds_in[t] = d_in[idx];

    for(int s = 1;s < BLOCK_SIZE;s*=2){
        __syncthreads();
        int operand = getIdentity(o);
        if(s <= t && t < BLOCK_SIZE && idx < size){
            operand = ds_in[t-s];
        }
        __syncthreads();
        process(&ds_in[t],&operand,o);
        // ds_in[t] += o;
    }
    __syncthreads();
    if(idx < size)
    d_in[idx] = ds_in[t];
}

// Sequential Scan/PrefixSum
void scanV0(int *A,int size,OPERATION o){
    int value = getIdentity(o);
    for(int i = 0;i < size;++i){ 
        process(&value,&A[i],o);
        A[i] = value;
    }
}

bool checkValid(int *A,int *B,int size){
    for(int i = 0;i < size;++i){
        if(A[i] != B[i])return 0;
    }
    return 1;
}

// Process each part p > 0
__global__ void parallelPartScan(int *d_in,int size,int number,OPERATION o){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size)return;
    process(&d_in[idx],&number,o);
    // d_in[idx] += number;
}

// Bleloch Algorithm (Exclusive Scan)
__global__ void scanV2(int *d_in,int size,OPERATION o){
    __shared__ int ds_in[BLOCK_SIZE];
    int t = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size)return;
    ds_in[t] = d_in[idx];
    __syncthreads();

    //UP SWEAP
    for(int s = 1;s < BLOCK_SIZE; s*=2){
        int i = 2 * s * (t+1) - 1;
        if(i < BLOCK_SIZE && i - s >= 0){
            // ds_in[i] += ds_in[i-s];
            process(&ds_in[i],&ds_in[i-s],o);
        }
        __syncthreads();
    }
    // DOWN SWEAP
    if(idx == size-1){
        ds_in[t] = getIdentity(o);
    }else{
        ds_in[BLOCK_SIZE - 1] = getIdentity(o);
    }

    for(int s = BLOCK_SIZE/2;s > 0;s/=2){
        int i = 2 * s * (t+1) - 1;

        if(i < BLOCK_SIZE && i - s >= 0){
            int leftChild = ds_in[i-s];
            ds_in[i-s] = ds_in[i];
            process(&ds_in[i],&leftChild,o);
            // ds_in[i] += leftChild;
        }
        __syncthreads();
    }


    if(idx < size) {
        //SHIFT VALUE
        if(idx == size-1){
            process(&d_in[idx],&ds_in[t],o);
            // d_in[idx] += ds_in[t];
        }else if(t == BLOCK_SIZE-1){
            process(&d_in[idx],&ds_in[t],o);
            // d_in[idx] += ds_in[t];
        }else{
            d_in[idx] = ds_in[t+1];
        }
    }

}

int main(){
    int n = N;
    int block_size = BLOCK_SIZE;
    int block = (n + block_size - 1) / block_size;
    OPERATION operation = SUM;

    dim3 dimGrid(block,1,1);
    dim3 dimBlock(block_size,1,1);


    int *h_in = new int[n];
    int *h_out = new int[n];
    int *check = new int[n];
    for(int i = 0;i < n;++i)h_in[i]=i+1;
    memcpy(check,h_in,n * sizeof(int));
    scanV0(check,n,operation);

    int *d_in;
    cudaMalloc((void**)&d_in,n * sizeof(int));
    cudaMemcpy(d_in,h_in,n * sizeof(int),cudaMemcpyHostToDevice);

    // scanV1<<<dimGrid,dimBlock>>>(d_in,n,operation);
    scanV2<<<dimGrid,dimBlock>>>(d_in,n,operation);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out,d_in,n*sizeof(int),cudaMemcpyDeviceToHost);

    for(int p = 1;p < block;++p){
        int start = block_size * p;
        int priorSum = h_out[start-1];

        int subN = (p == block - 1) ? (n - start) : block_size;
        int subBlockSize = block_size;
        int subBlock = (subN + subBlockSize - 1) / subBlockSize;
        cudaMemcpy(d_in,h_out+start,subN * sizeof(int),cudaMemcpyHostToDevice);
        parallelPartScan<<<subBlock,subBlockSize>>>(d_in,subN,priorSum,operation);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out+start,d_in,subN * sizeof(int),cudaMemcpyDeviceToHost);
        
        // for(int i = 0; i < block_size;++i){
        //     if(start + i < n)
        //     h_out[start + i] += priorSum;
        // }
    }

    for(int i = 0;i < n;++i){
        cout << "OUT[" << i << "]" << h_out[i] << endl;
    }
    cout << "VALID : " << (checkValid(h_out,check,n) ? "TRUE" : "FALSE" ) << endl;




}