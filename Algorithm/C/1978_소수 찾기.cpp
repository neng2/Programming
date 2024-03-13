#include <iostream>
using namespace std;
int arr[1001];
int n;
int cnt;
int main() {
	cin >> n;
	for (int i = 2; i <= 1000; i++) {
		arr[i] = i;
	}
	for (int i = 2; i <= 1000; i++) {
		for (int j = 2; j <= 1000; j++) {
			if (arr[j] != i && arr[j] % i == 0) {
				arr[j] = 0;
			}
		}
	}
	for (int i = 0; i < n; i++) {
		int temp;
		cin >> temp;
		if (arr[temp] != 0)cnt++;
	}
	cout << cnt;
}
/*
수학
정수론
소수 판정
*/