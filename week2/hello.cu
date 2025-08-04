#include <iostream>

__global__ void hello(){
    printf("Hello Parallel World!! from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}       

int main(){
    hello<<<3, 33>>>();
    cudaDeviceSynchronize();
}