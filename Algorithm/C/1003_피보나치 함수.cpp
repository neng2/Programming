#include <stdio.h>
#include <malloc.h>
int main() {
	int T;
	int *arr;
	int n[41][2];
	scanf("%d", &T);
	arr = (int*)malloc(4 * T);
	for (int i = 0; i < T; i++) {
		scanf("%d", &arr[i]);
	}
	n[0][0] = 1;
	n[0][1] = 0;
	n[1][0] = 0;
	n[1][1] = 1;
	for (int i = 2; i < 41; i++) {
		n[i][0] = n[i - 1][0] + n[i - 2][0];
		n[i][1] = n[i - 1][1] + n[i - 2][1];
	}
	for (int i = 0; i < T; i++) {
		printf("%d %d\n",n[arr[i]][0],n[arr[i]][1]);
	}
}
/*
다이나믹 프로그래밍
*/