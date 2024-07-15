#include <stdio.h>
#include <math.h>
#define ll long long
#define MAX_DIGIT 6

int n;
int main() {
	for (; scanf("%d", &n) != -1;) {
		long long num[251][MAX_DIGIT] = { 0, };
		num[0][0] = 1;
		num[1][0] = 1;
		//num[2][0] = 3;
		for (int i = 2; i <= n; i++) {
			for (int j = 0; j < MAX_DIGIT ; j++) {
				//num[i][j + 1] = num[i - 1][j + 1] + num[i - 2][j + 1] * 2;
				if ((num[i][j] + num[i - 1][j] + (num[i - 2][j] * 2)) % (ll)pow(10, 18)
					!= num[i][j] + num[i - 1][j] + num[i - 2][j] * 2) {
					num[i][j + 1] = (num[i][j] + num[i - 1][j] + num[i - 2][j] * 2) / (ll)pow(10, 18);
					num[i][j] = (num[i][j]+num[i - 1][j] + num[i - 2][j] * 2) % (ll)pow(10, 18);
				}
				else {
					num[i][j] = num[i][j]+num[i - 1][j] + num[i - 2][j] * 2;
				}

			}
		}
		int flag = 1;
		for (int j = MAX_DIGIT - 1; j >= 0; j--) {
			if (num[n][j] == 0 && flag)continue;
			else if (num[n][j] != 0 && flag) {
				flag = 0;
				printf("%lld", num[n][j]);
			}
			else if (!flag) {
				for (int i = 17; i >= 0; i--) {
					printf("%lld", num[n][j] / (ll)pow(10, i));
					num[n][j] = num[n][j] % (ll)pow(10, i);
				}
			}
		}
		printf("\n");
	}
}

/*
다이나믹 프로그래밍
임의 정밀도 / 큰 수 연산
*/