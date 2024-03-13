#include <stdio.h>

int main() {
	int N[436], M[436], dp[436][30];
	int T,cnt=1;
	scanf("%d", &T);
	for (int i = 1; i <= T; i++) {
		scanf("%d %d", &N[i], &M[i]);
		if (N[i] == M[i]) {
			dp[i][(int)N[i]] = 1;
			continue;
		}
		dp[i][1] = M[i];
		for (int j = 2; j <= N[i]; j++) {
			dp[i][j] = (dp[i][j - 1] * (M[i] - j + 1)) / j;
		}
	}
	for (int i = 1; i <= T; i++) {
		printf("%d\n", dp[i][N[i]]);
	}
}
/*
수학
다이나믹 프로그래밍
조합론
*/