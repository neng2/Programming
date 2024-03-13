#include <stdio.h>
#define FOR(a,b) for(int i=a;i<=b;i++)

int N;
int T[16], P[16], result[30], val;

int main() {
	scanf("%d", &N);
	FOR(1, N) {
		scanf("%d %d", &T[i], &P[i]);
	}

	FOR(1, N) {
		if (i + T[i] <= N + 1) {
			result[i + T[i]] = result[i + T[i]] > result[i] + P[i] ? result[i + T[i]] : result[i] + P[i];
			if (result[i + T[i]] > val)val = result[i + T[i]];
		}
		result[i + 1] = result[i + 1] > result[i] ? result[i + 1] : result[i];
		if (result[i + 1] > val)val = result[i + 1];

	}
	printf("%d", val);
}
/*
다이나믹 프로그래밍
브루트포스 알고리즘
*/