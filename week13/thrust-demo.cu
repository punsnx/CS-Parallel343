#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace std;

int main(){
    int n = 10;
    thrust::host_vector<long> a(n);
    for(int i = 0;i < n;++i){
        a[i] = i;
    }

    thrust::device_vector<long> a(a);
    thrust::











    return 0;
}

