#include <iostream>
using namespace std;
int n;
int result = 1;
int main() {
	cin >> n;
	for (int i = 1; i <= n; i++) {
		result *= i;
	}
	cout << result;
}
/*
수학
구현
조합론
*/