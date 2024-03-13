#include <iostream>
using namespace std;
int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);
	int cnt = 0, n, org;
	cin >> n;
	org = n;
	do {
		cnt++;
		n = (n % 10) * 10 + ((n / 10 + n % 10) % 10);
	} while (n != org);
	cout << cnt;
}
/*
수학
구현
*/