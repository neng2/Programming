#include <iostream>
#include <algorithm>
using namespace std;
int n;
int arr[1000];
int main() {
	cin >> n;
	int min = 1;
	for (int i = 0; i < n; i++) {
		cin >> arr[i];
	}
	sort(arr, arr + n);
	for (int i = 0; i < n; i++) {
		if (arr[i] > min)break;
		else min += arr[i];
	}
	cout << min;
}

/*
그리디 알고리즘
정렬
*/