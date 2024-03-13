#include <stdio.h>

int dp[5001],n;
int MIN(int a, int b){
    if(a>b)return b;
    else return a;
}
int MAX(int a,int b){
    if(a>b)return a;
    else return b;
}
int main(){
    scanf("%d",&n);
    dp[3]=1;
    dp[5]=1;
    for(int i=6;i<=n;i++){
        int a = MIN(dp[i-3],dp[i-5]);
        int b = MAX(dp[i-3],dp[i-5]);
        if(a){
            dp[i] = a+1;
        }
        else if(b){
            dp[i] = b+1;
        }
    }
    if(dp[n]!=0)printf("%d",dp[n]);
    else printf("-1");
}
/*
수학
다이나믹 프로그래밍
그리디 알고리즘
*/