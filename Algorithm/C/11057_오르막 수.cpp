#include <stdio.h>

int dp[1001][10], sum, n;
int main() {
	scanf("%d", &n);
	for (int i = 0; i <= 9; i++) {
		dp[1][i] = 1;
		sum += dp[1][i];
	}
	for (int i = 2; i <= n; i++) {
		sum = dp[i][9] = 1;
		for (int j = 0; j <= 8; j++) {
			for (int k = j; k <= 9; k++) {
				dp[i][j] += dp[i-1][k];
				dp[i][j] %= 10007;
			}
			if (i == n) {
				sum += dp[i][j];
			}
		}
	}
	printf("%d", sum%10007);
}
/*
다이나믹 프로그래밍
*/