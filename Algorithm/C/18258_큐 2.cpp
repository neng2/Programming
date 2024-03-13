#include <iostream>

#include <string>
using namespace std;

typedef struct NODE {
	int data;
	NODE *ptr_prev;
}NODE;

typedef struct QUEUE {
	int len; 
	NODE *top;
	NODE *bottom;
}QUEUE;

int main() {
	ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0);
	int n;
	QUEUE *a = new QUEUE;
	a->len = 0;
	a->bottom = NULL;
	a->top = NULL;

	string command;

	cin >> n;
	cin.ignore(); // 버퍼 비우기
	for (int i = 0; i < n; i++) {
		getline(cin, command);
		if (command.find("push") == 0) {
			int len = command.size();
			char *num = new char[len-5];
			command.copy(num, len - 5, 5);// push+" " 까지의 문자열 길이가 5이므로 숫자만 복사하기 위해서
			int datanumber = atoi(num);
			NODE *newNode = new NODE;
			newNode->data = datanumber;
			newNode->ptr_prev = NULL;
			if (a->top == NULL){
				a->top = newNode;
				a->bottom = newNode;
			}
			else {
				a->bottom->ptr_prev = newNode;
				a->bottom = newNode;
			}
			a->len++;
		}
		else if (command == "pop") {
			if (a->top == NULL) {
				cout << -1 << "\n";
			}
			else {
				cout << a->top->data << "\n";
				a->top = a->top->ptr_prev; 
				a->len--;
			}
		}
		else if (command == "size") {
			cout << a->len << "\n";
		}
		else if (command == "empty") {
			if (a->top == NULL) {
				cout << 1 << "\n";
			}
			else cout << 0 << "\n";
		}
		else if (command == "front") {
			if (a->top == NULL) {
				cout << -1 << "\n";
			}
			else cout << a->top->data << "\n";
		}
		else if (command == "back") {
			if (a->top == NULL) {
				cout << -1 << "\n";
			}
			else cout << a->bottom->data << "\n";
		}
	}
}
/*
자료 구조
큐
*/