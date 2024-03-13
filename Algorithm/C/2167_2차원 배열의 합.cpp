#include <stdio.h>

long long int sum[10001];
int main() {
	int arr[301][301], n, m, k, i, j, x, y;
	scanf("%d %d", &n, &m);
	//n = m = 300;
	for (int p = 1; p <= n; p++) {
		for (int q = 1; q <= m; q++) {
			scanf("%d", &arr[p][q]);
			//arr[p][q] = 10000;
		}
	}
	scanf("%d", &k);
	//k = 10000;
	for (int p = 1; p <= k; p++) {
		scanf("%d %d %d %d", &i, &j, &x, &y);
		//i = j = 1;
		//x = y = 300;
		for (int q = i; q <= x; q++) {
			for (int r = j; r <= y; r++) {
				sum[p] += arr[q][r];
			}
		}
	}
	for (int p = 1; p <= k; p++) {
		printf("%lld\n", sum[p]);
	}
}
/*
구현
누적 합
*/