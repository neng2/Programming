#include <stdio.h>
#include <malloc.h>
#define MIN(a,b) (a < b) ? a : b
int main() {
	int n, *arr;
	scanf("%d", &n);
	arr = (int*)malloc(sizeof(int)*n+1);
	arr[1] = 0;
	for (int i = 2; i <= n; i++) {
		arr[i] = arr[i-1]+1;
		if (i % 2 == 0)arr[i] = MIN(arr[i],arr[i / 2] + 1);
		if (i % 3 == 0)arr[i] = MIN(arr[i],arr[i / 3] + 1);
	}
	printf("%d", arr[n]);
}
/*
다이나믹 프로그래밍
*/