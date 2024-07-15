#include <iostream>
using namespace std;
int n,m;
int map[1025][1025];
int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);
	cin >> n >> m;
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= n; j++) {
			int temp;
			cin >> temp;
			if (i == 1 && j == 1)map[i][j] = temp;
			else if (i == 1) {
				map[i][j] = map[i][j - 1] + temp;
			}
			else if (i == 1) {
				map[i][j] = map[i - 1][j] + temp;
			}
			else {
				map[i][j] = map[i][j - 1] + map[i - 1][j] - map[i - 1][j - 1] + temp;
			}
		}
	}
	for (int i = 0; i < m; i++) {
		int x1, y1, x2, y2;
		cin >> x1 >> y1 >> x2 >> y2;
		int sum = map[x2][y2] - map[x1 - 1][y2] - map[x2][y1 - 1] + map[x1 - 1][y1 - 1];
		cout << sum << "\n";
	}
}
/*
다이나믹 프로그래밍
누적 합
*/