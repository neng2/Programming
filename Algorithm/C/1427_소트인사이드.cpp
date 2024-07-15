#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;
int arr[11];
int n;
bool cmpr(int a, int b) {
	return a > b;
}
int main() {
	cin >> n;
	int size = (int)log10(n) + 1;
	int i = 0;
	while (n) {
		arr[i] = n % 10;
		n /= 10;
		i++;
	}
	sort(arr, arr + size,cmpr);
	for (int i = 0; i < size; i++) {
		cout << arr[i];
	}
}
/*
문자열
정렬
*/