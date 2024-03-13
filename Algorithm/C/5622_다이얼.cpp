#include <iostream>
#include <string.h>
using namespace std;
char str[16];
int time;
int main() {
	cin >> str;
	for (int i = 0; i < strlen(str); i++) {
		if (str[i] <= 'O') {
			time += (str[i] - 'A') / 3 + 3;
		}
		else if (str[i] <= 'S')time += 8;
		else if (str[i] <= 'V')time += 9;
		else time += 10;
	}
	cout << time;
}

/*
구현
*/