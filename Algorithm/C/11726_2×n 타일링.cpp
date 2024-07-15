#include <stdio.h>
int main() {
	int n[1001];
	int N;
	n[0] = 1;
	n[1] = 1;
	scanf("%d", &N);
	for (int i = 2; i <= N; i++) {
		n[i] = (n[i - 1] + n[i - 2]) % 10007;
	}
	printf("%d\n", n[N]);
}

/*
다이나믹 프로그래밍
*/