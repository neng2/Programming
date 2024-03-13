#include <iostream>
#include <string>
#include <stack>
#include <sstream>
using namespace std;
int main(){
	int n;
	cin >> n;
	cin.ignore();
	for(int i=0;i<n;i++){
		string str;
		getline(cin,str);
		stringstream token(str);
		cout << "Case #" << i+1 << ": ";
		stack <string> word;
		string temp;
		while(token>>temp){
			word.push(temp);
		}
		while(!word.empty()){
			cout << word.top() << " ";
			word.pop();
		}
		cout<< endl;
	}
}
/*
자료 구조
문자열
파싱
스택
*/