#include <iostream>
#include <string>
using namespace std;
int main() {
	int n;
	int sum = 0;
	char num[101];
	cin >> n;
	cin.ignore();
	for (int i = 0; i < n; i++) {
		cin >> num[i];
		sum += num[i] - '0';
	}
	printf("%d", sum);
}
/*
수학
구현
문자열
*/