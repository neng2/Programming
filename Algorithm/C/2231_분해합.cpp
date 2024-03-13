#include <iostream>
using namespace std;

int n;
int main() {
	cin >> n;
	int result = 0;
	for (int i = 1; i <= 56; i++) {
		int temp = n - i;
		int sum = 0;
		while (temp) {
			sum += temp % 10;
			temp /= 10;

		}
		if (sum == i)result = n - i;
	}
	cout << result;
}
	
/*
브루트포스 알고리즘
*/