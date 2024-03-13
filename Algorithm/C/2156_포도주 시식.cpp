#include <stdio.h>
int MAX(int a, int b) {
	return a > b ? a : b;
}
int main() {
	int wine[10001], sum[10001][4], n, cnt = 1;
	scanf("%d", &n);
	for (int i = 1; i <= n; i++) {
		scanf("%d", &wine[i]);
	}
	int _max = 0;
	sum[1][0] = sum[1][1] = sum[1][2] = wine[1];
	sum[2][1] = wine[1] + wine[2];
	sum[2][2] = sum[2][3] = wine[2];
	sum[3][1] = wine[2] + wine[3];
	sum[3][2] = wine[1] + wine[3];
	sum[3][3] = wine[3];
	for (int i = 4; i <= n; i++) {
		sum[i][1] = MAX(sum[i - 1][2], sum[i - 1][3]) + wine[i];
		sum[i][2] = MAX(sum[i - 2][1], sum[i - 2][2]) + wine[i];
		sum[i][3] = sum[i - 3][1] + wine[i];
	}
	for (int i = 1; i <= n; i++) {
		if (sum[i][1] > _max)_max = sum[i][1];
		if (sum[i][2] > _max)_max = sum[i][2];
		if (sum[i][3] > _max)_max = sum[i][3];
	}
	printf("%d\n", _max);
}

/*
다이나믹 프로그래밍
*/