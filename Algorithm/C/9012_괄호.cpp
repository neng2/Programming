#include <iostream>
#include <stack>
#include <string>
using namespace std;

int n, result;
int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);
	cin >> n;
	cin.ignore();
	for (int i = 0; i < n; i++) {
		result = 1;
		stack<char> s;
		string Parenthesis;
		getline(cin, Parenthesis);
		for (int j = 0; j < Parenthesis.size(); j++) {
			if (Parenthesis[j] == '(') {
				s.push(Parenthesis[j]);
			}
			else if (Parenthesis[j] == ')'&&s.size()) {
				s.pop();
			}
			else result = 0;
		}
		if (s.size())result = 0;
		if (result) {
			cout << "YES\n";
		}
		else {
			cout << "NO\n";
		}
	}
}

/*
자료 구조
문자열
스택
*/