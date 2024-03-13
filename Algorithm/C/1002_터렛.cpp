#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;
int n;
int point;
int main() {
	cin >> n;
	while (n) {
		double x1, x2, y1, y2, r1, r2;
		double d;
		cin >> x1 >> y1 >> r1 >> x2 >> y2 >> r2;
		d = sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
		if (d == r1 + r2 || d == abs(r1 - r2))point = 1;
		else if (d > abs(r1 - r2) && d < r1 + r2)point = 2;
		else point = 0;

		if (x1 == x2 && y1 == y2 && r1 == r2)point = -1;
		cout << point << endl;
		n--;
	}
}
/*
수학
기하학
많은 조건 분기
*/