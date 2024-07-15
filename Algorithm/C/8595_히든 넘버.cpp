#include <iostream>
#include <string>
#include <cctype>

#define ll long long
using namespace std;
int main(){
    ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    ll n , sum = 0;
    string str;
    string cache="";
    char prev;
    cin >> n;
    cin >> str;
    prev = str[0];
    for(int i=0; i < n; i++){
        if(isdigit(str[i])&&!isdigit(prev)){
            cache = "";
            cache += str[i];
        }
        else if(isdigit(str[i])&&isdigit(prev)){
            cache += str[i];
        }
        else if(!isdigit(str[i])&&isdigit(prev)){
            int temp = stoi(cache);
            if(temp>999999)temp=0;
            sum += temp;
            cache="";
        }
        prev = str[i];
    }
    if(cache!=""){
        int temp = stoi(cache);
        if(temp>999999)temp=0;
        sum += temp;
    }
    cout << sum;
}
/*
문자열
파싱
*/