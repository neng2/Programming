#include <iostream>
using namespace std;
int main(){
    int chess[6]={1,1,2,2,2,8},input[6],sum[6]={0,};
    for(int i=0;i<6;i++){
        cin>>input[i];
        chess[i]-=input[i];
        cout<<chess[i]<<" ";
    }
}
/*
수학
구현
사칙연산
*/