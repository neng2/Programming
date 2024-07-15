#include <stdio.h>

int main() {
	long long int P[101];
	int T,t[100001];
	scanf("%d", &T);
	P[1] = P[2] = P[3] = 1;
	P[4] = P[5] = 2;
	for (int i = 6; i <= 100; i++) {
		P[i] = P[i - 1] + P[i - 5];
	}
	for (int i = 1; i <= T; i++) {
		scanf("%d", &t[i]);
	}
	for (int i = 1; i <= T; i++) {
		printf("%lld\n", P[t[i]]);
	}
}
/*
수학
다이나믹 프로그래밍
*/