#include <stdio.h>

int main() {
	int num[12],N,tc[11];
	num[1] = 1;
	num[2] = 2;
	num[3] = 4;
	scanf("%d", &N);
	for (int i = 0; i < N; i++) {
		scanf("%d", &tc[i]);
	}
	for (int i = 4; i <= 11; i++) {
		num[i] = num[i - 1] + num[i - 2] + num[i - 3];
	}
	for (int i = 0; i < N; i++) {
		printf("%d\n", num[tc[i]]);
	}
}
/*
다이나믹 프로그래밍
*/