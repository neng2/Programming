#include <stdio.h>
#define MIN(a,b) ((a<b) ? a:b)
int main() {
	int n, arr[1001][3];
	scanf("%d", &n);
	for (int i = 1; i <= n; i++) {
		for (int j = 0; j < 3; j++) {
			scanf("%d", &arr[i][j]);
		}
	}
	for (int i = 2; i <= n; i++) {
			arr[i][0] += MIN(arr[i-1][1], arr[i-1][2]);
			arr[i][1] += MIN(arr[i-1][0], arr[i-1][2]);
			arr[i][2] += MIN(arr[i-1][0], arr[i-1][1]);
	}
	printf("%d", MIN(MIN(arr[n][0], arr[n][1]), arr[n][2]));
}

/*
다이나믹 프로그래밍
*/