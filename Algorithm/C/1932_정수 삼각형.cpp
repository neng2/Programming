#include <stdio.h>
int _max(int a, int b) {
	if (a > b)return a;
	else return b;

}
int main() {
	int num[501][501];
	int sum[501][501];
	int N, max;
	scanf("%d", &N);
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= i; j++) {
			scanf("%d", &num[i][j]);
			sum[i][j] = num[i][j];
		}
	}
	for (int i = 2; i <= N; i++) {
		for (int j = 1; j <= i; j++) {
			if (j == 1) {
				sum[i][j] = sum[i][j] + sum[i - 1][j];
			}
			else if (j == i) {
				sum[i][j] = sum[i][j] + sum[i - 1][j - 1];
			}
			else sum[i][j] = sum[i][j] + _max(sum[i - 1][j - 1], sum[i - 1][j]);
		}
	}
	max = sum[N][1];
	for (int i = 2; i <= N; i++) {
		max = _max(max, sum[N][i]);
	}
	printf("%d", max);
}

/*
다이나믹 프로그래밍
*/