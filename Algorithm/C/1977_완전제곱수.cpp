#include <iostream>
using namespace std;
int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);
	int M, N, first = 0;
	int min_, sum = 0;
	cin >> M >> N;
	for (int i = 1; i*i <= N; i++) {
		int temp = i * i;
		if (temp < M)continue;
		if (!first) {
			min_ = temp;
			first = 1;
		}
		sum += temp;
	}
	if(first)cout << sum << "\n" << min_;
	else cout << -1;
}

/*
수학
구현
브루트포스 알고리즘
*/