#include <iostream>
using namespace std;
int a, b;
int main() {
	int cnt = 1;
	cin >> a >> b;
	while (a != b) {
		if (a > b) {
			cnt = -1;
			break;
		}
		if ((b % 10) % 2 == 0) {
			b /= 2;
			cnt++;
		}
		else if (b % 10 == 1) {
			b--;
			cnt++;
			b /= 10;
		}
		else {
			cnt = -1;
			break;
		}
	}
	cout << cnt;
}

/*
그래프 이론
그리디 알고리즘
그래프 탐색
너비 우선 탐색
*/