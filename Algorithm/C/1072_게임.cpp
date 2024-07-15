#include <stdio.h>

int main() {
	long long int X, Y,Z , a;
	while (scanf("%lld %lld", &X, &Y) != EOF) {
		Z = 100 * Y / X;
		if (Z<99) {
			a = (X*(1+Z) - 100*Y)/(99-Z);
			if ((X*(1 + Z) - 100 * Y) % (99 - Z)!=0) {
				a++;
			}
		}
		else a = -1;

		printf("%lld\n", a);
	}
}

/*
수학
이분 탐색
*/