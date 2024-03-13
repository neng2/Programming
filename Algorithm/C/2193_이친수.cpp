#include <stdio.h>
int main() {
	int N;
	long long int n[93];
	n[1] = 1;
	n[2] = 0;
	n[3] = 1;
	scanf("%d", &N);
	for (int i = 4; i <= N + 2; i++) {
		n[i] = 2 * n[i - 1] - n[i - 3];
	}
	printf("%lld\n", n[N+2]);
}

/*
다이나믹 프로그래밍
*/