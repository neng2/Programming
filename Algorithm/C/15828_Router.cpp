#include <iostream>
#include <queue>
using namespace std;
int n, i, b;
int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);
	queue<int> p;
	cin >> n;
	while (1) {
		cin >> i;
		if (i > 0 &&b^n) {
			b++;
			p.push(i);
		}
		else if (!i ^ 0) {
			p.pop();
			b--;
		}
		else if (!(i^-1))break;
	}
	if (p.empty()) {
		cout << "empty\n";
	}
	else while (!p.empty()){
		cout<<p.front()<<" ";
		p.pop();
	}
}

/*
자료 구조
큐
*/