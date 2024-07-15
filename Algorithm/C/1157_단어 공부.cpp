#include <iostream>
#include <string.h>
using namespace std;
char str[1000001];
int check[100];
int _max = -1;
char result;
int main() {
	cin >> str;
	int prev_ = 0;
	int length = strlen(str);
	for (int i = 0; i < length; i++) {
		if (str[i] <= 'z'&&str[i] >= 'a')str[i] -= 32;
		check[str[i]]++;
	}
	for (int i = 'A'; i <= 'Z'; i++) {
		if (check[i] > _max) {
			_max = check[i];
			result = i;
		}
		else if (check[i] == _max) {
			result = '?';
		}
	}
	cout << result;
}
/*
구현
문자열
*/