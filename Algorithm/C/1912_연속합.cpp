#include <stdio.h>

int _max(int a, int b) {
	if (a > b)return a;
	else return b;
}
int main() {
	int num[100001], ans[100001] ,max,sub;
	int n;
	scanf("%d", &n);
	for (int i = 1; i <= n; i++) {
		scanf("%d", &num[i]);
	}
	max = ans[1] = num[1];
	for (int i = 2; i <= n; i++) {
		ans[i] = _max(ans[i - 1] + num[i], num[i]);
		if (ans[i] > max)max = ans[i];
	}
	printf("%d", max);
}
/*
다이나믹 프로그래밍
*/