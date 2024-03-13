#include <iostream>
#include <algorithm>
using namespace std;

#define ll long long int
ll result=0;
pair<int, int> line[1000001];
int N;
int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);
	cin >> N;
	for (int i = 0; i < N; i++) {
		cin >> line[i].first>>line[i].second;
	}
	sort(line, line + N);
	ll t_front = -2000000000, t_back = -2000000000;
	for (int i = 0; i < N; i++) {
		if (t_front <= line[i].first && t_back >= line[i].second)continue;
		result += (line[i].second - line[i].first);
		if (t_back > line[i].first)result -= (t_back - line[i].first);
		t_front = line[i].first;
		t_back = line[i].second;
	}
	cout << result;
}
/*
정렬
스위핑
*/