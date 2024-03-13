#include <iostream>
#include <algorithm>
using namespace std;
int arr[9];
int n, m;

void dfs(int a,int *print, int depth) {
	int visit[8] = { 0, };
	if (depth == 0) {
		for (int i = m - 1; i >= 0; i--) {
			cout << print[i] << " ";
		}
		cout << "\n";
		return;
	}
	for (int i = a; i < n; i++) {
		print[depth-1] = arr[i];
		dfs(i, print,depth-1);
	}
}

int main() {
	int cnt;
	int print[8] = { 0, };
	int visit[8] = { 0, };
	cin >> n >> m;
	for (int i = 0; i < n; i++) {
		cin >> arr[i];
	}
	sort(arr, arr + n);
	cnt = m;
	dfs(0, print, m);
}
/*
백트래킹
*/