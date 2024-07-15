#include <stdio.h>
#include <algorithm>
int MIN(int a, int b){
    if(a>b)return b;
    else return a;
}

int main(){
    int coin[101],dp[10001]={0,},n,k,min;
    scanf("%d %d",&n,&k);
    for(int i=1;i<=n;i++){
        scanf("%d",&coin[i]);
    }
    std::sort(coin , coin+n);
    dp[0]=0;
    for(int i=1;i<=k;i++){
        dp[i]=-1;
        min=99999999;
        for(int j=1;j<=n;j++){
            if(i-coin[j]>=0&&dp[i-coin[j]]!=-1){
                min = MIN(min,dp[i-coin[j]]);
            }
        }
        dp[i]=min+1;
    }
    if(dp[k]<99999999)printf("%d",dp[k]);
    else printf("-1");
}

/*
다이나믹 프로그래밍
*/