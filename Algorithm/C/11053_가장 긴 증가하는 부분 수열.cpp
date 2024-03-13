#include <stdio.h>
int MAX(int a, int b) {
	if (a > b)return a;
	else return b;
}
int main() {
	int dp[1001], n, num[1001], max = 1;
	scanf("%d", &n);
	for (int i = 1; i <= n; i++) {
		scanf("%d", &num[i]);
		dp[i] = 1;
	}
	for (int i = 2; i <= n; i++) {
		for (int j = 1; j < i; j++) {
			if (num[j] < num[i]) {
				dp[i] = MAX(dp[j] + 1, dp[i]);
			}
		}
		if (dp[i] > max)max = dp[i];
	}
	printf("%d\n", max);
}
/*
다이나믹 프로그래밍
*/