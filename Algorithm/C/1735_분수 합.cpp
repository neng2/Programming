#include <iostream>
using namespace std;

int a1, a2;
int b1, b2;
int gcd(int a, int b) {
	while (b > 0) {
		int temp = b;
		b = a % b;
		a = temp;
	}
	return a;
}
int main() {
	int gcd_;
	cin >> a1>>a2;
	cin >> b1 >> b2;
	gcd_ = gcd(a2, b2);
	int lcm = (a2*b2) / gcd_;
	int c1 = a1 * (lcm / a2) + b1 * (lcm / b2);
	gcd_ = gcd(c1, lcm);
	cout << c1 / gcd_ << " " << lcm / gcd_;
}

/*
수학
정수론
유클리드 호제법
*/