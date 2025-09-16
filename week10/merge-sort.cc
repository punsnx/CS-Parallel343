#include <iostream>
#include <cstring>
using namespace std;

void merge(int *arr,int s,int m,int e){
    int n1 = m - s + 1;
    int n2 = e - m;
    
    int i = 0, j = 0;
    int k = s;
    int *L = new int[n1];
    int *R = new int[n2];
    memcpy(L,&arr[s],n1 * sizeof(int));
    memcpy(R,&arr[m+1],n2 * sizeof(int));
    
    while(i < n1 && j < n2){
        if(L[i] <= R[j]){
            arr[k] = L[i++];
        }else{
            arr[k] = R[j++];
        }
        ++k;
    }
    
    while(i < n1){
        arr[k++] = L[i++];
    }
    while(j < n2){
        arr[k++] = R[j++];
    }
    delete[] L;
    delete[] R;
}

void mergeSort(int* arr,int start,int end){
    if(start >= end)return;
    int mid = (end + start)/2;

    mergeSort(arr,start,mid);
    mergeSort(arr,mid+1,end);
    
    merge(arr,start,mid,end);
    // 0 1 2 3 4 5 6 7 8 9
}
int main(){
    
    int arr[10] = {1,4,6,3,8,7,2,5,9,0};
    mergeSort(arr,0,9);
    for(int i = 0;i < 10;++i){
        cout << arr[i] << " "; 
    }
    cout << endl;

    return 0;
}