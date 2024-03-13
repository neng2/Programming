#include <iostream>
#include <algorithm>
using namespace std;
int n, m;
int arr[100001];

void bs(int n, int temp) {
	int left = 0;
	int right = n - 1;
	int mid;
	for (; right - left >= 0;) {
		mid = (left + right) / 2;
		if (arr[mid] == temp) {
			cout << 1<<"\n";
			return;
		}
		else if (arr[mid] > temp) {
			right = mid - 1;
		}
		else {
			left = mid + 1;
		}
	}
	cout << 0 << "\n";
	return;
}


int main() {
	ios_base::sync_with_stdio; cin.tie(0); cout.tie(0);
	cin >> n;
	for (int i = 0; i < n; i++) {
		cin >> arr[i];
	}
	cin >> m;
	sort(arr, arr + n);
	int temp;
	for (int i = 0; i < m; i++) {
		cin >> temp;
		bs(n, temp);
	}
}

/*
자료 구조
정렬
이분 탐색
*/