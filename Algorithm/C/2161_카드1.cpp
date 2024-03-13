#include <iostream>
#include <queue>

using namespace std;

int n, i, b;
int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);
	queue<int> p;
	queue<int> q;
	cin >> n;
	for (int i = 1; i <= n; i++) {
		p.push(i);
	}
	while (p.size()-1) {
		i++;
		if (i % 2) {
			q.push(p.front());
			p.pop();
		}
		else {
			p.push(p.front());
			p.pop();
		}
	}
	while (q.size()) {
		cout << q.front() << " ";
		q.pop();
	}
	cout << p.front();
}
/*
구현
자료 구조
큐
*/