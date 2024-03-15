#include <iostream>
#define fastio ios::sync_with_stdio(0), cin.tie(0), cout.tie(0)
using namespace std;
char str[1048575*4];
int i=0;
void hanoi(int N, int source, int dest){
    if(N>1){
        int remain=6-source-dest;
        hanoi(N-1,source,remain);
        str[i++]=source+'0';
        str[i++]=' ';
        str[i++]=dest+'0';
        str[i++]='\n';
        hanoi(N-1,remain,dest);
    }
    else {
        str[i++]=source+'0';
        str[i++]=' ';
        str[i++]=dest+'0';
        str[i++]='\n';//입출력시간 줄이기위해 문자열로 만들고 출력2
    }
}

int main(){
    fastio;
    int N; cin>>N;
    hanoi(N,1,3);
    str[i]='\0';
    cout<<(1<<N)-1<<"\n"<<str;
}
/*
재귀.
*/