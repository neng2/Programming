#include <stdio.h>

int MAX(int a, int b) {
	if (a > b)return a;
	else return b;
}

int main() {
	int P[1001],dp[1001], n;
	scanf("%d", &n);
	for (int i = 1; i <= n; i++) {
		scanf("%d", &P[i]);
		dp[i] = P[i];
	}
	for (int i = 2; i <= n; i++) {
		for (int j = 1; j <= i / 2; j++) {
			dp[i] = MAX(dp[i - j] + dp[j], dp[i]);
		}
	}
	printf("%d\n", dp[n]);
}
/*
다이나믹 프로그래밍
*/