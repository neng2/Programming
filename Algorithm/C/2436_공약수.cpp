#include <iostream>
#include <cmath>
#define ll long long
using namespace std;

int main(){
    ll G, L;
    ll ans[2];
    ll min=9223372036854775807;
    ll mn, sqrt_mn;
    cin >> G >> L;
    mn = L/G;
    sqrt_mn = (int)sqrt(mn);
    for(int i = 1; i <= sqrt_mn ; i++){
        if(mn%i)continue;
        else{
            int a=i, b=mn/i;
            while(b){
                int temp = a%b;
                a=b;
                b=temp;
            }
            if(a!=1)continue;
            if(( i*G + (mn/i) * G) < min){
                min = i*G + (mn/i) * G;
                ans[0] = i*G;
                ans[1] = (mn/i) * G;
            }
        }
    }
    cout << ans[0] << " " << ans[1];
}

/*
수학
브루트포스 알고리즘
정수론
유클리드 호제법
*/