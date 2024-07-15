#include <stdio.h>
#define FOR(a,b) for(int i=a;i<=b;i++)

int n;
long long int num[91];
int main() {
	scanf("%d", &n);
	num[0] = 0;
	num[1] = 1;
	FOR(2, n) {
		num[i] = num[i - 1] + num[i - 2];
	}
	printf("%lld\n", num[n]);
}


/*
수학
다이나믹 프로그래밍
*/