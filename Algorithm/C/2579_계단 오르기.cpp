#include <stdio.h>
inline int MAX(int a, int b) {
	return a > b ? a : b;
}
int main() {
	int stair[301], sum[301][3], n, cnt = 1;
	scanf("%d", &n);
	for (int i = 1; i <= n; i++) {
		scanf("%d", &stair[i]);
	}
	sum[1][0] = stair[1];
	sum[1][1] = stair[1];
	sum[1][2] = stair[1];
	sum[2][1] = stair[1] + stair[2];
	sum[2][2] = stair[2];
	for (int i = 3; i <= n; i++) {
		sum[i][1] = sum[i-1][2] + stair[i];
		sum[i][2] = MAX(sum[i-2][1] + stair[i], sum[i - 2][2] + stair[i]);
	}
	printf("%d", MAX(sum[n][1], sum[n][2]));
}

/*
다이나믹 프로그래밍
*/