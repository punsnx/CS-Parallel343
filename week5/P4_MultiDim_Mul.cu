#include <stdio.h>

void matrixMul2DV0(int *h1_in,int *h2_in,int *h_out, int n){
    for(int r = 0;r < n; ++r){
        for(int c = 0; c < n;++c){
            h_out[r * n + c] += h1_in[r * n + c] * h2_in[c * n + r];
        }
    }
        int *h3_out = (int *)calloc(n*n,sizeof(int));
    matrixMul2DV0(h1_in,h2_in,h3_out,n);



}


int main(){

    //matrix multiplication

    int *h3_out = (int *)calloc(n*n,sizeof(int));
    matrixMul2DV0(h1_in,h2_in,h3_out,n);


    for(int r = 0;r<n;++r){
        for(int c = 0;c<n;++c){
            printf("%d ",h3_out[r * n + c]);
        }
        printf("\n");
    }

}
