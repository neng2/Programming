#include <stdio.h>
#include <math.h>
int main() {
	int N;
	int sum = 0;
	int r;
	scanf("%d", &N);
	r = (int)log10(N);
	for (int i = 1; i <= r; i++) {
		sum += i * 9 * pow(10, i - 1);
	}
	sum += (N - pow(10, r) + 1)*(r+1);
	printf("%d", sum);
}
/*
수학
구현
*/