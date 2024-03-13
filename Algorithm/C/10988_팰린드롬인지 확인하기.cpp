#include <iostream>
#include <cstring>
using namespace std;
int main(  ){
    char word[101]={0,};
    cin>>word;
    int palindrome=1,i,j;
    int length=strlen(word);
    if(length%2){
         for(i=length/2,j=i;j>=0;i++,j--){
            if(word[i]!=word[j])palindrome=0;
         }
    }
    else{
         for(i=length/2,j=i-1;j>=0;i++,j--){
             if(word[i]!=word[j])palindrome=0;
         }
    }
    cout<<palindrome;
}

/*
문자열
브루트포스 알고리즘
*/