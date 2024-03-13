#include <iostream>

using namespace std;

int map[502][502];
int map_height[502][502];
int M, N;
void compare_height(int i, int j) {
	int result = 0;
	if (map_height[i][j] < map_height[i][j - 1]) {
		result += map[i][j - 1];
	}if (map_height[i][j] < map_height[i][j + 1]) {
		result += map[i][j + 1];
	}if (map_height[i][j] < map_height[i - 1][j]) {
		result += map[i - 1][j];
	}if (map_height[i][j] < map_height[i + 1][j]) {
		result += map[i + 1][j];
	}
	map[i][j] = result;
}
int main() {
	cin >> M >> N;
	map[0][1] = 1;
	for (int i = 1; i <= M; i++) {
		map_height[0][i] = 10001;
		map_height[i][0] = 10001;
		map_height[501][i] = 10001;
		map_height[i][501] = 10001;
		for (int j = 1; j <= N; j++) {
			cin >> map_height[i][j];
		}
	}
	for (int cnt = 1; cnt <= 5; cnt++) {
		for (int i = 1; i <= M; i++) {
			for (int j = 1; j <= N; j++) {
				compare_height(i, j);
			}
		}
	}
	cout << map[M][N];
	
}

/*
다이나믹 프로그래밍
그래프 이론
그래프 탐색
깊이 우선 탐색
*/