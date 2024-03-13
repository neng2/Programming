#include <stdio.h>

int main() {
	int num[100][10], sum = 0, n;
	scanf("%d", &n);
	num[0][0] = 0;
	for (int i = 1; i <= 9; i++) {
		num[0][i] = 1;
	}
	for (int i = 1; i < n; i++) {
		num[i][0] = num[i - 1][1];
		num[i][9] = num[i - 1][8];
		for (int j = 1; j <= 8; j++) {
			num[i][j] = (num[i - 1][j - 1] + num[i - 1][j + 1]) % 1000000000;
		}
	}
	for (int i = 0; i < 10; i++) {
		sum += num[n-1][i];
		sum %= 1000000000;
	}
	printf("%d\n", sum);
}
/*
다이나믹 프로그래밍
*/