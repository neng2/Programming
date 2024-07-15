#include <stdio.h>
#include <algorithm>
using namespace std;
int main() {
	int n, k,coin[101],ans = 0, dp[10001] = { 0, };
	scanf("%d %d", &n, &k);
	for (int i = 1; i <= n; i++) {
		scanf("%d", &coin[i]);
	}
	sort(&coin[1], &coin[n]+1);
	dp[0] = 1;
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= k; j++) {
			if (j - coin[i] >= 0) {
				dp[j] += dp[j - coin[i]];
			}
		}
	}
	printf("%d", dp[k]);
}

/*
다이나믹 프로그래밍
*/